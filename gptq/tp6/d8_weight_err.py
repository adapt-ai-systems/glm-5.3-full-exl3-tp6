"""D8: data-free (RTN) 4-bit weight error on real GLM 5.3 BF16 non-routed linears.

Input: data/<tensor>.bin (raw BF16) + data/index.json (name -> [N, K]).
Schemes (all per output row, groups along K):
  mxfp8      current D2 (vLLM _mxfp8_e4m3_quantize_torch, E8M0 block 32)   8.25 b/w
  int4s_g128 GPTQ-style uint4b8 symmetric RTN, BF16 scale (vLLM quantize_weights) 4.125
  int4s_g64 / int4s_g32                                                       4.25 / 4.5
  int4a_g128 AWQ-style uint4 + zero point RTN                                 4.156
  int4s_g128_clip / int4a_g128_clip  + per-group MSE-optimal clip (data-free)
  nvfp4_g16  FP4 E2M1, E4M3 group-16 scale + FP32 global scale                4.5
  mxfp4_g32  FP4 E2M1, E8M0 group-32 scale (OCP)                              4.25
Metric: ||W - Q(W)||_F / ||W||_F per tensor (= output rel. err for isotropic x).
"""
import ast
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(14)
D = Path(sys.argv[1] if len(sys.argv) > 1 else 'data')
CH = 1024  # rows per chunk


def _load_mxfp8():
    src = (Path(__file__).parent / 'mxfp8_utils.py').read_text()
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in (
        '_mxfp8_e4m3_quantize_torch', 'dequant_mxfp8_to_bf16')]
    ns = {'torch': torch, 'MXFP8_BLOCK_SIZE': 32, 'MXFP8_VALUE_DTYPE': torch.float8_e4m3fn,
          'MXFP8_SCALE_DTYPE': torch.uint8, 'swizzle_mxfp8_scale': None}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'mxfp8_utils', 'exec'), ns)
    return ns['_mxfp8_e4m3_quantize_torch'], ns['dequant_mxfp8_to_bf16']


QMX, DQMX = _load_mxfp8()


def mxfp8(w):
    q, s = QMX(w.to(torch.bfloat16).contiguous(), False)
    return DQMX(q, s.view(w.shape[0], -1)).float()


def _bf16(x):
    return x.to(torch.bfloat16).float()


def int4(w, g, asym=False, clip=False):
    n, k = w.shape
    x = w.view(n, k // g, g)
    ratios = torch.linspace(1.0, 0.5, 26) if clip else torch.tensor([1.0])
    best, best_err = None, None
    mx, mn = x.amax(-1, keepdim=True), x.amin(-1, keepdim=True)
    for r in ratios:
        if asym:
            hi, lo = mx * r, mn * r
            s = _bf16(((hi - lo).clamp(min=1e-5) / 15))
            zp = torch.round((-lo / s).abs()).clamp(0, 15)  # vLLM: round(|min/s|)
            q = (torch.round(x / s) + zp).clamp(0, 15)
            y = (q - zp) * s
        else:
            s = _bf16(torch.maximum(mx * r / 7, -mn * r / 8).clamp(min=1e-12))
            y = torch.round(x / s).clamp(-8, 7) * s
        err = ((y - x) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = y, err
        else:
            m = err < best_err
            best = torch.where(m, y, best)
            best_err = torch.where(m, err, best_err)
    return best.view(n, k)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _fp4(v):
    """Nearest E2M1 (ties to even code), sign kept, saturate at 6."""
    a = v.abs().clamp(max=6.0)
    mids = (E2M1[1:] + E2M1[:-1]) / 2
    idx = torch.bucketize(a, mids)  # ties go to lower bucket
    tie = torch.isin(a, mids)
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)  # odd code at tie -> even
    return E2M1[idx.clamp(max=7)] * torch.sign(v)


def nvfp4(w, gscale):
    n, k = w.shape
    x = w.view(n, k // 16, 16)
    amax = x.abs().amax(-1, keepdim=True)
    s = (amax / 6 / gscale).to(torch.float8_e4m3fn).float() * gscale
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (_fp4(x / s) * s).view(n, k)


def mxfp4(w):
    n, k = w.shape
    x = w.view(n, k // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp(min=2 ** -126)
    s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    return (_fp4(x / s) * s).view(n, k)


SCHEMES = {
    'mxfp8': (8.25, lambda w, ctx: mxfp8(w)),
    'int4s_g128': (4.125, lambda w, ctx: int4(w, 128)),
    'int4s_g64': (4.25, lambda w, ctx: int4(w, 64)),
    'int4s_g32': (4.5, lambda w, ctx: int4(w, 32)),
    'int4a_g128': (4.156, lambda w, ctx: int4(w, 128, asym=True)),
    'int4s_g128_clip': (4.125, lambda w, ctx: int4(w, 128, clip=True)),
    'int4a_g128_clip': (4.156, lambda w, ctx: int4(w, 128, asym=True, clip=True)),
    'nvfp4_g16': (4.5, lambda w, ctx: nvfp4(w, ctx['gscale'])),
    'mxfp4_g32': (4.25, lambda w, ctx: mxfp4(w)),
}


def main():
    idx = json.loads((D / 'index.json').read_text())
    rows = []
    for name, (n, k) in idx.items():
        w = torch.frombuffer(bytearray((D / (name + '.bin')).read_bytes()),
                             dtype=torch.bfloat16).view(n, k)
        ctx = {'gscale': w.float().abs().max().item() / (6 * 448)}
        wn2 = w.float().pow(2).sum().item()
        r = {'name': name, 'N': n, 'K': k, 'params': n * k}
        for sch, (_, fn) in SCHEMES.items():
            e2 = 0.0
            for i in range(0, n, CH):
                wf = w[i:i + CH].float()
                e2 += (fn(wf, ctx) - wf).pow(2).sum().item()
            r[sch] = round((e2 / wn2) ** 0.5, 5)
        rows.append(r)
        print(json.dumps(r), flush=True)
    tot = sum(r['params'] for r in rows)
    agg = {s: round((sum(r[s] ** 2 * r['params'] for r in rows) / tot) ** 0.5, 5) for s in SCHEMES}
    print(json.dumps({'summary': 'param-weighted rms rel err', 'params': tot,
                      'bits_per_weight': {s: b for s, (b, _) in SCHEMES.items()}, **agg}))


if __name__ == '__main__':
    main()
