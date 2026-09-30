#!/usr/bin/env python3
"""TP4 GPTQ int4 solve for the D8 targets. The numerics are the TP6 solver's, imported unchanged
(gptq/tp6/solve_gptq.py: gptq_codes / quant_group_scale / pack / fingerprint /
proxy_err); only the rank geometry and the Hessian sources are TP4-specific.

TP6 solve_gptq.py:34 hardcodes TP=6, HPR=12, SR=384 (shared inter padded 2048->2304), 72-head padded o_proj.
Native TP4 needs no padding: 64 heads -> 16/rank, shared inter 2048 -> 512/rank, dense inter 12288 -> 3072/rank.
Deployed per-rank slices (vLLM TP4 column/row parallel; checked against the TP4 capture's fingerprints):
  fused_qkv_a   q_a rows [2048, 6144], replicated          o_proj       [:, r*4096:(r+1)*4096]
  shared gate_up  gate/up rows r*512..(r+1)*512            shared down  [:, r*512:(r+1)*512]
  dense gate_up   gate/up rows r*3072..                    dense down   [:, r*3072:(r+1)*3072]

Hessians:
  row-parallel inputs (o_proj, down_proj): --cap  = the TP4 capture, r<rank>/<prefix>.pt (per rank, TP4-only)
  replicated inputs (fused_qkv_a, gate_up): --cap-rep (default: same as --cap) at r<layer % --rep-tp> (or r0).
     These inputs are the full hidden state, identical in content under any TP split, so the TP6 capture
     (--cap-rep <tp6 capture dir> --rep-tp 6, same calib_ids.jsonl) is valid and saves ~23 GB of copying.
  fingerprints (slice check) always from --cap r<rank>/fingerprints.pt (the TP4 deployed weights).
Outlier re-solve inline (was fix_outliers.py): if dequant-vs-BF16 weight rel err > --maxw (0.2, the loader's hard
limit), re-solve that module with percdamp 0.03 -> 0.1 -> 0.3 -> 1.0 and keep the first under the limit.
q_a (replicated) codes are cached in <out>/shared/ so per-rank runs (--ranks 0, then 1, ...) solve it once.
Output: <out>/r<rank>/layer-LLL.safetensors, same format + metadata as solve_gptq.py (loader: d2_fp8_gptq.py).
Usage: solve_gptq_tp4.py --dense D --cap C --out O [--cap-rep C6 --rep-tp 6] [--ranks 0-3] [--group 32]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
TP6_SOLVER = Path(os.environ.get('TP6_SOLVER', HERE.parent / 'tp6' / 'solve_gptq.py'))
PIN = HERE / 'tp6_solver.sha256'


def _load_tp6():
    data = TP6_SOLVER.read_bytes()
    want = PIN.read_text().split()[0]
    assert hashlib.sha256(data).hexdigest() == want, f'{TP6_SOLVER} changed (want {want})'
    import sys
    sys.path.insert(0, str(TP6_SOLVER.parent))  # it imports d8_weight_err (RTN proxy) from its own dir
    spec = importlib.util.spec_from_file_location('solve_gptq_tp6', TP6_SOLVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load_tp6()
TP, DV, HEADS, SHARED_INTER, DENSE_INTER = 4, 256, 64, 2048, 12288
HPR, SR, DR = HEADS // TP, SHARED_INTER // TP, DENSE_INTER // TP  # 16, 512, 3072


def rank_weights(f, layer, r, tp=TP):
    """prefix -> deployed 4-bit rows [n4, k] (float32) for TP rank r of a tp-way split (tp=6 reproduces TP6)."""
    if tp == 6:
        return S.rank_weights(f, layer, r)
    hpr, sr, dr = HEADS // tp, SHARED_INTER // tp, DENSE_INTER // tp
    g = lambda n: f.get_tensor(f'model.layers.{layer}.{n}').float()
    p = f'model.layers.{layer}.'
    o = g('self_attn.o_proj.weight')
    assert o.shape[1] == HEADS * DV, o.shape
    W = {p + 'self_attn.fused_qkv_a_proj': g('self_attn.q_a_proj.weight'),
         p + 'self_attn.o_proj': o[:, r * hpr * DV:(r + 1) * hpr * DV]}
    if layer < 3:
        a, b = r * dr, (r + 1) * dr
        W[p + 'mlp.gate_up_proj'] = torch.cat([g('mlp.gate_proj.weight')[a:b], g('mlp.up_proj.weight')[a:b]])
        W[p + 'mlp.down_proj'] = g('mlp.down_proj.weight')[:, a:b]
    else:
        a, b = r * sr, (r + 1) * sr
        W[p + 'mlp.shared_experts.gate_up_proj'] = torch.cat([g('mlp.shared_experts.gate_proj.weight')[a:b],
                                                              g('mlp.shared_experts.up_proj.weight')[a:b]])
        W[p + 'mlp.shared_experts.down_proj'] = g('mlp.shared_experts.down_proj.weight')[:, a:b]
    return {k: v.contiguous() for k, v in W.items()}


def load_h(cap, prefix, ranks):
    for r in ranks:
        path = cap / f'r{r}' / f'{prefix}.pt'
        if path.exists():
            c = torch.load(path)
            if c['H'] is not None and c['n'] > 0:
                return c['H'].float() / c['n'], c['n'], r
    raise FileNotFoundError(f'no H for {prefix} in {cap} r{ranks}')


def relw(W, C, Sc, g):
    q = S.dequant(C, Sc.to(torch.bfloat16).float(), g)
    return ((q - W).norm() / W.norm()).item()


def solve_module(W, Hs, g, clip, maxw):
    C, Sc = S.gptq_codes(W, Hs, g, clip)
    e0 = e = relw(W, C, Sc, g)
    damp = 0.01
    if e > maxw:
        for d in (0.03, 0.1, 0.3, 1.0):
            C2, S2 = S.gptq_codes(W, Hs, g, clip, percdamp=d)
            e2 = relw(W, C2, S2, g)
            if e2 < e:
                C, Sc, e, damp = C2, S2, e2, d
            if e2 <= maxw:
                break
    return C, Sc, e0, e, damp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True); ap.add_argument('--cap', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cap-rep', default=None); ap.add_argument('--rep-tp', type=int, default=TP)
    ap.add_argument('--layers', default='0-77'); ap.add_argument('--ranks', default='0-3')
    ap.add_argument('--group', type=int, default=32); ap.add_argument('--no-clip', action='store_true')
    ap.add_argument('--maxw', type=float, default=0.2)
    ap.add_argument('--no-fp-check', action='store_true'); ap.add_argument('--threads', type=int, default=14)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dense, cap, out = Path(a.dense), Path(a.cap), Path(a.out)
    cap_rep = Path(a.cap_rep) if a.cap_rep else cap
    g, clip = a.group, not a.no_clip
    ranks = S.parse_range(a.ranks)
    assert all(0 <= r < TP for r in ranks), ranks
    fpc = {}
    (out / 'shared').mkdir(parents=True, exist_ok=True)
    for layer in S.parse_range(a.layers):
        dests = {r: out / f'r{r}' / f'layer-{layer:03d}.safetensors' for r in ranks}
        if not a.force and all(d.exists() for d in dests.values()):
            continue
        t0 = time.time()
        res = {r: {} for r in ranks}
        with safe_open(dense / f'layer-{layer:03d}.safetensors', 'pt') as f:
            for r in ranks:
                for prefix, W in rank_weights(f, layer, r).items():
                    rel = None if a.no_fp_check else S.fp_check(cap, r, prefix, W, fpc)
                    rep = S.replicated(prefix)
                    shared = out / 'shared' / f'{prefix}.pt'
                    if rep and prefix.endswith('fused_qkv_a_proj') and shared.exists():
                        c = torch.load(shared)
                        C, Sc, ev = c['C'], c['S'], dict(event='reuse', layer=layer, rank=r, module='self_attn.fused_qkv_a_proj')
                    else:
                        if rep:
                            Hs, ncal, hr = load_h(cap_rep, prefix, [layer % a.rep_tp, 0])
                            src = f'rep r{hr}'
                        else:
                            Hs, ncal, hr = load_h(cap, prefix, [r])
                            src = f'r{hr}'
                        t1 = time.time()
                        C, Sc, e0, e1, damp = solve_module(W, Hs, g, clip, a.maxw)
                        ev = dict(event='solve', layer=layer, rank=r, module=prefix.split('.', 3)[3], n=W.shape[0],
                                  k=W.shape[1], ncal=ncal, h_from=src, fp_rel=rel, relw=round(e0, 4),
                                  relw_final=round(e1, 4), percdamp=damp,
                                  proxy_err_gptq=round(S.proxy_err(W, S.dequant(C, Sc, g), Hs), 5),
                                  s=round(time.time() - t1, 1))
                        if e1 > a.maxw:
                            ev['event'] = 'solve_over_maxw'  # the boot loader would hard-fail this module
                        if rep and prefix.endswith('fused_qkv_a_proj'):
                            torch.save({'C': C, 'S': Sc}, str(shared) + '.part')
                            Path(str(shared) + '.part').rename(shared)
                    print(json.dumps(ev), flush=True)
                    res[r][prefix + '.q4'] = S.pack(C)
                    res[r][prefix + '.s4'] = Sc.to(torch.bfloat16)
                    res[r][prefix + '.fp'] = S.fingerprint(W)
        meta = dict(group=str(g), clip=str(int(clip)), damp='0.01+outlier', fp_check=str(int(not a.no_fp_check)),
                    layer=str(layer), tp='4')
        for r, d in dests.items():
            d.parent.mkdir(parents=True, exist_ok=True)
            save_file(res[r], str(d) + '.part', metadata=dict(meta, rank=str(r)))
            Path(str(d) + '.part').rename(d)
        print(json.dumps(dict(event='layer', layer=layer, ranks=ranks, s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(event='done', ranks=ranks)), flush=True)


if __name__ == '__main__':
    main()
