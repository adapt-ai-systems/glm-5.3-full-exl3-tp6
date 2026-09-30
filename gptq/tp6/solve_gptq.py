"""D8-GPTQ offline solve (CPU, any large-RAM x86 host): GPTQ int4 (symmetric uint4b8, group G, per-group MSE clip) codes +
scales for every D8 target module on every TP6 rank, from the capture boot's Hessians and the BF16 dense weights.

Inputs
  --dense DIR   dense/layer-LLL.safetensors (extract_dense.py: q_a, o_proj, [shared_experts.]gate/up/down)
  --cap DIR     capture dump: r<rank>/<prefix>.pt {H = sum x^T x, n} (+ r<rank>/fingerprints.pt)
                replicated-input modules (fused_qkv_a, gate_up) live on r<layer % 6> (or r0 for layers 3/40/77)
Output
  --out DIR     r<rank>/layer-LLL.safetensors, per module prefix:
                  <prefix>.q4 uint8 [n4, k/2]   codes 0..15 (= q + 8), low nibble = even column
                  <prefix>.s4 bf16  [n4, k/G]   group scales
                  <prefix>.fp fp32  [n4]        fingerprint of the deployed BF16 rows (loader re-checks at boot)
Deployed per-rank slices (vLLM TP6 loader; checked against the capture's fingerprints of the real loaded weights):
  fused_qkv_a  q_a rows [2048, 6144], replicated          o_proj  [:, r*3072:(r+1)*3072] of the 72-head padded
  shared gate_up  gate/up rows r*384.. of 2304-padded      shared down  [:, r*384..] of 2304-padded
  dense (0-2) gate_up  gate/up rows r*2048..               dense down  [:, r*2048..]
Per module the log has GPTQ vs RTN-clip proxy output error sqrt(tr(D H D^T) / tr(W H W^T)) on the calibration H.
Usage: solve_gptq.py --dense D --cap C --out O [--layers 0-77] [--ranks 0-5] [--group 64] [--no-fp-check]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d8_weight_err as WE  # noqa: E402

TP, HPR, DV, SR, DR_INTER = 6, 12, 256, 384, 2048
FP_TOL = 1e-3


def fingerprint(w):
    """Same as the capture overlay / loader: fp32 [n] = w @ v, v a fixed integer hash in [-0.5, 0.5)."""
    k = w.shape[1]
    v = ((torch.arange(k, dtype=torch.int64, device=w.device) * 2654435761) % 1009).float() / 1009 - 0.5
    return w.float() @ v


def pad_rows(w, n):
    return torch.cat([w, w.new_zeros(n - w.shape[0], w.shape[1])]) if w.shape[0] < n else w


def pad_cols(w, k):
    return torch.cat([w, w.new_zeros(w.shape[0], k - w.shape[1])], 1) if w.shape[1] < k else w


def rank_weights(f, layer, r):
    """prefix -> deployed 4-bit rows [n4, k] (BF16 values, float32) for rank r."""
    g = lambda n: f.get_tensor(f'model.layers.{layer}.{n}').float()
    p = f'model.layers.{layer}.'
    o = pad_cols(g('self_attn.o_proj.weight'), 72 * DV)
    W = {p + 'self_attn.fused_qkv_a_proj': g('self_attn.q_a_proj.weight'),
         p + 'self_attn.o_proj': o[:, r * HPR * DV:(r + 1) * HPR * DV]}
    if layer < 3:
        a, b = r * DR_INTER, (r + 1) * DR_INTER
        W[p + 'mlp.gate_up_proj'] = torch.cat([g('mlp.gate_proj.weight')[a:b], g('mlp.up_proj.weight')[a:b]])
        W[p + 'mlp.down_proj'] = g('mlp.down_proj.weight')[:, a:b]
    else:
        a, b = r * SR, (r + 1) * SR
        sg, su = pad_rows(g('mlp.shared_experts.gate_proj.weight'), TP * SR), pad_rows(g('mlp.shared_experts.up_proj.weight'), TP * SR)
        W[p + 'mlp.shared_experts.gate_up_proj'] = torch.cat([sg[a:b], su[a:b]])
        W[p + 'mlp.shared_experts.down_proj'] = pad_cols(g('mlp.shared_experts.down_proj.weight'), TP * SR)[:, a:b]
    return {k: v.contiguous() for k, v in W.items()}


def replicated(prefix):
    return prefix.endswith(('fused_qkv_a_proj', 'gate_up_proj'))


def quant_group_scale(w, clip):
    """w [n, g] -> bf16-representable per-row scale [n] (symmetric [-8, 7]); MSE clip search over 26 ratios."""
    mx, mn = w.amax(-1, keepdim=True), w.amin(-1, keepdim=True)
    ratios = torch.linspace(1.0, 0.5, 26).tolist() if clip else [1.0]
    best_s = best_err = None
    for rt in ratios:
        s = torch.maximum(mx * rt / 7, -mn * rt / 8).clamp(min=1e-12).to(torch.bfloat16).float()
        err = (torch.round(w / s).clamp(-8, 7) * s - w).pow(2).sum(-1, keepdim=True)
        if best_s is None:
            best_s, best_err = s, err
        else:
            m = err < best_err
            best_s, best_err = torch.where(m, s, best_s), torch.where(m, err, best_err)
    return best_s[:, 0]


def gptq_codes(W, Hs, g, clip, block=128, percdamp=0.01):
    """GPTQ as layer0_study.gptq (no act-order, static groups at group start from the error-updated weights; identical
    when no input column is dead, damping from live columns otherwise), returning codes int8 [n, k] in [-8, 7] and scales float32 [n, k/g]; dequant = codes * scales."""
    W = W.float().clone()
    n, k = W.shape
    Hm = Hs.clone()
    dead = torch.diag(Hm) == 0
    # damping from the LIVE diagonal only: TP padding (rank 5 o_proj heads 64-71, shared down cols 2048-2303) gives dead
    # columns; averaging their placeholder diag into the damping swamps small real activations (GPTQ -> ~RTN)
    damp = percdamp * torch.diag(Hm)[~dead].mean() if (~dead).any() else torch.tensor(1.0)
    Hm[dead, dead] = damp.clamp(min=1e-12)
    W[:, dead] = 0
    Hm += damp * torch.eye(k)
    L = torch.linalg.cholesky(Hm)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    C = torch.zeros(n, k, dtype=torch.int8)
    S = torch.zeros(n, k // g)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        W1, E1, Hi = W[:, i1:i2].clone(), torch.zeros(n, i2 - i1), Hinv[i1:i2, i1:i2]
        s = None
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:
                s = quant_group_scale(W1[:, i:i + g], clip)
                S[:, (i1 + i) // g] = s
            w = W1[:, i]
            c = torch.round(w / s).clamp(-8, 7)
            C[:, i1 + i] = c.to(torch.int8)
            e = (w - c * s) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return C, S


def dequant(C, S, g):
    n, k = C.shape
    return (C.float().view(n, k // g, g) * S.unsqueeze(-1)).view(n, k)


def pack(C):
    u = (C.to(torch.int16) + 8).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


def unpack(P):
    return torch.stack([P & 15, P >> 4], -1).view(P.shape[0], -1).to(torch.int8) - 8


def proxy_err(W, Wq, Hs):
    D = W - Wq
    return ((D @ Hs * D).sum() / (W @ Hs * W).sum()).clamp(min=0).sqrt().item()


def load_cap(cap, prefix, ranks):
    for r in ranks:
        p = cap / f'r{r}' / f'{prefix}.pt'
        if p.exists():
            c = torch.load(p)
            if c['H'] is not None and c['n'] > 0:
                return (c['H'].float() / c['n']), c['n'], r
    raise FileNotFoundError(f'no H for {prefix} in {cap} r{ranks}')


def fp_check(cap, r, prefix, w, cache):
    if r not in cache:
        p = cap / f'r{r}' / 'fingerprints.pt'
        cache[r] = torch.load(p) if p.exists() else None
    fps = cache[r]
    if fps is None or prefix not in fps:
        raise RuntimeError(f'no captured fingerprint for r{r} {prefix}')
    ref, shape = fps[prefix]
    ref = ref[:w.shape[0]].float()
    rel = ((fingerprint(w) - ref).norm() / ref.norm().clamp(min=1e-12)).item()
    if not (rel <= FP_TOL and shape[1] == w.shape[1] and shape[0] >= w.shape[0]):
        raise RuntimeError(f'slice mismatch r{r} {prefix}: fp rel {rel:.3g}, captured shape {shape}, ours {tuple(w.shape)}')
    return rel


def parse_range(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True); ap.add_argument('--cap', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='0-77'); ap.add_argument('--ranks', default='0-5')
    ap.add_argument('--group', type=int, default=64); ap.add_argument('--no-clip', action='store_true')
    ap.add_argument('--no-fp-check', action='store_true'); ap.add_argument('--threads', type=int, default=14)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dense, cap, out = Path(a.dense), Path(a.cap), Path(a.out)
    g, clip = a.group, not a.no_clip
    ranks = parse_range(a.ranks)
    fpc = {}
    for layer in parse_range(a.layers):
        dests = {r: out / f'r{r}' / f'layer-{layer:03d}.safetensors' for r in ranks}
        if not a.force and all(d.exists() for d in dests.values()):
            continue
        t0 = time.time()
        res = {r: {} for r in ranks}
        solved = {}  # replicated modules: one solve (same weight + same H on every rank)
        with safe_open(dense / f'layer-{layer:03d}.safetensors', 'pt') as f:
            for r in ranks:
                for prefix, W in rank_weights(f, layer, r).items():
                    rel = None if a.no_fp_check else fp_check(cap, r, prefix, W, fpc)
                    rep = replicated(prefix)
                    key = (prefix, None if rep and prefix.endswith('fused_qkv_a_proj') else r)
                    if key not in solved:
                        Hs, ncal, hr = load_cap(cap, prefix, [layer % TP, 0] if rep else [r])
                        t1 = time.time()
                        C, S = gptq_codes(W, Hs, g, clip)
                        e_g = proxy_err(W, dequant(C, S, g), Hs)
                        e_r = proxy_err(W, WE.int4(W, g, clip=True), Hs)
                        solved[key] = (C, S)
                        print(json.dumps(dict(event='solve', layer=layer, rank=r, module=prefix.split('.', 3)[3],
                                              n=W.shape[0], k=W.shape[1], ncal=ncal, h_from=f'r{hr}', fp_rel=rel,
                                              proxy_err_gptq=round(e_g, 5), proxy_err_rtn=round(e_r, 5),
                                              s=round(time.time() - t1, 1))), flush=True)
                    C, S = solved[key]
                    res[r][prefix + '.q4'] = pack(C)
                    res[r][prefix + '.s4'] = S.to(torch.bfloat16)
                    res[r][prefix + '.fp'] = fingerprint(W)
        meta = dict(group=str(g), clip=str(int(clip)), damp='0.01', fp_check=str(int(not a.no_fp_check)), layer=str(layer))
        for r, d in dests.items():
            d.parent.mkdir(parents=True, exist_ok=True)
            save_file(res[r], str(d) + '.part', metadata=dict(meta, rank=str(r)))
            Path(str(d) + '.part').rename(d)
        print(json.dumps(dict(event='layer', layer=layer, s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(event='done')), flush=True)


if __name__ == '__main__':
    main()
