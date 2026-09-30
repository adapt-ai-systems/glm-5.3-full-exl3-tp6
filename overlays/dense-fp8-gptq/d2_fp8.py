"""D2: MXFP8 (FP8 E4M3 weights, E8M0 block-32 scales) Marlin W8A16 for the
BF16 non-routed linears of GLM 5.3 TP6, plus the lm_head (shared by the MTP
drafter).

Off unless GLM_D2_FP8=1. Weights load as BF16 through the normal loaders; each
module is converted in process_weights_after_loading and its BF16 copy freed,
so no loader or checkpoint change is needed.

Env:
  GLM_D2_FP8=1            enable
  GLM_D2_FP8_EXCLUDE=re   prefixes to keep BF16 (default below)
  GLM_D2_FP8_HEAD=0       keep lm_head BF16
  GLM_D2_FP8_LOG=1        log every converted module
  GLM_D2_FP8_DEQUANT_M=N  rows >= N: dequantize the FP8 weight into one shared
                          BF16 scratch and use the BF16 GEMM (prefill); rows < N:
                          Marlin (decode). 0 = Marlin only. Default 256.
  GLM_D2_FP8_BIG=w8a8|bf16  large-M path. w8a8 (D2c, default): Marlin bytes ->
                          plain FP8 scratch + MXFP8 activations -> FlashInfer
                          CUTLASS SM120 MXFP8 GEMM. bf16 (D2b): dequant + BF16 GEMM.
                          w8a8 falls back to bf16 if its load-time check fails.
  GLM_D2_FP8_W8A8_TUNE=0  skip the load-time GEMM tactic pick (tactic 0)
  GLM_D2_FP8_W8A8_TOL=x   max rel. error of the W8A8 load check (default 0.08)
"""
import logging
import os
import re

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

log = logging.getLogger('vllm.glm_d2_fp8')

# kv_b_proj: MLA absorbs it into W_UK/W_UV at load time (dequant path untested).
# mlp.gate: router logits. shared_head.head: MTP copy, replaced by the target
# lm_head after loading (and empty in this checkpoint).
DEFAULT_EXCLUDE = r'.*(\.kv_b_proj|\.mlp\.gate|shared_head\.head)$'
BLOCK = 32
_stats = {'modules': 0, 'bf16_bytes': 0, 'fp8_bytes': 0}


def enabled() -> bool:
    return os.environ.get('GLM_D2_FP8', '0') == '1'


def eligible(prefix: str, is_lm_head: bool) -> bool:
    if not enabled():
        return False
    if is_lm_head and os.environ.get('GLM_D2_FP8_HEAD', '1') != '1':
        return False
    exclude = os.environ.get('GLM_D2_FP8_EXCLUDE', DEFAULT_EXCLUDE)
    return re.fullmatch(exclude, prefix) is None


def quantize_mxfp8(w: torch.Tensor, rows: int = 2048):
    """Row-chunked copy of vLLM's torch MXFP8 quantizer (bit-identical per row).

    Chunking bounds the FP32 temporaries (lm_head is 25856 x 6144 per rank).
    The FlashInfer quantizer vLLM picks on cc>=100 is a cute-dsl SM100 path,
    so it is not used on GB10.
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )
    n, k = w.shape
    assert k % BLOCK == 0, (n, k)
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((n, k // BLOCK), dtype=torch.uint8, device=w.device)
    for i in range(0, n, rows):
        qi, si = _mxfp8_e4m3_quantize_torch(w[i:i + rows].contiguous(), False)
        q[i:i + rows].copy_(qi)
        s[i:i + rows].copy_(si.view(qi.shape[0], -1))
    return q, s


def _convert(layer: torch.nn.Module, prefix: str) -> bool:
    """Replace layer.weight (BF16 [N, K]) with Marlin MXFP8 weight + scales."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.utils import replace_parameter

    w = layer.weight.data
    if w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        return False
    n, k = w.shape
    if k % BLOCK or n < 64:
        log.warning('GLM D2 FP8: keeping %s BF16 (shape %s)', prefix, tuple(w.shape))
        return False
    dtype = w.dtype
    q, s = quantize_mxfp8(w)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.input_scale = None
    replace_parameter(layer, 'weight', q)
    replace_parameter(layer, 'weight_scale', s)
    del w
    kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # Marlin scale permutation uses the default dtype
    try:
        kernel.process_weights_after_loading(layer)
    finally:
        torch.set_default_dtype(prev)
    layer._glm_d2_kernel = kernel
    layer._glm_d2_deq = _setup_dequant(layer, prefix, q, s)
    del q, s
    _stats['modules'] += 1
    _stats['bf16_bytes'] += n * k * 2
    _stats['fp8_bytes'] += n * k + n * k // BLOCK
    if n * k >= 64 << 20:
        torch.cuda.empty_cache()
    if os.environ.get('GLM_D2_FP8_LOG', '0') == '1':
        log.info('GLM D2 FP8: %s [%d, %d] -> MXFP8 Marlin', prefix, n, k)
    if _stats['modules'] % 50 == 0 or prefix.endswith('lm_head'):
        log.info('GLM D2 FP8 running total: %d modules, %.2f GB BF16 -> %.2f GB FP8',
                 _stats['modules'], _stats['bf16_bytes'] / 1e9,
                 _stats['fp8_bytes'] / 1e9)
    return True


def _apply(layer, x, bias):
    if DISPATCH:
        return torch.ops.vllm.glm_d2_mxfp8_linear(
            x, layer.weight, layer.weight_scale, layer.workspace, bias,
            layer.output_size_per_partition, layer.input_size_per_partition,
            layer._glm_d2_deq)
    return layer._glm_d2_kernel.apply_weights(layer, x, bias)


# ---------------------------------------------------------------------------
# D2b: M-threshold dispatch. The branch lives inside one opaque custom op, so
# the compiled graph is the same for every M and CUDA-graph capture (decode
# sizes 5..20) records only the Marlin path. At M >= DEQUANT_M a Triton kernel
# rebuilds the plain BF16 [N, K] weight from the Marlin-packed FP8 bytes and
# E8M0 scales (no extra copy of the weights is kept) into one scratch buffer
# sized for the largest layer, then cuBLAS runs the BF16 GEMM.
# ---------------------------------------------------------------------------
DEQUANT_M = int(os.environ.get('GLM_D2_FP8_DEQUANT_M', '256'))
DISPATCH = DEQUANT_M > 0
_deq = {'ok': DISPATCH, 'scratch': None, 'tables': {}, 'checked': 0, 'op': False, 'max_nk': 0}
BN, BK = 64, 128
BIG = os.environ.get('GLM_D2_FP8_BIG', 'w8a8')
if BIG not in ('w8a8', 'bf16'):
    raise ValueError('GLM_D2_FP8_BIG must be w8a8 or bf16, got %r' % BIG)
W8A8_TOL = float(os.environ.get('GLM_D2_FP8_W8A8_TOL', '0.08'))
TUNE_MS = (128, 256, 512, 1024, 2048)
# D2c state: one plain-FP8 weight scratch + one swizzled-scale scratch (largest
# layer), the FlashInfer module + workspace, per-(N, K) tactic tables.
_w8 = {'ok': DISPATCH and BIG == 'w8a8', 'w': None, 's': None, 'ws': None, 'mod': None,
       'quant': None, 'tactics': {}, 'checked': set()}


def marlin_weight_perm_8bit() -> list:
    """vLLM get_weight_perm(num_bits=8) (marlin_utils_test.py), as a list."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    inter = (0, 2, 1, 3)
    return [perm[4 * a + inter[b]] for a in range(len(perm) // 4) for b in range(4)]


def marlin_scale_perm() -> list:
    """vLLM get_scale_perms()[0] (group_size < K)."""
    return [i + 8 * j for i in range(8) for j in range(8)]


def dequant_tables_cpu():
    """(inv weight perm [1024], scale map [64]) as int32 CPU tensors.

    Weight: original (k, n) sits at byte (k//16)*16N + (c//1024)*1024 + inv[c%1024]
    of the Marlin tensor, c = (n//16)*256 + (k%16)*16 + n%16.
    Scale: original flat (k//32)*N + n = q sits at (q//64)*64 + smap[q%64] of the
    Marlin scale tensor (scale perm, then the [0,2,1,3] e8m0 swizzle).
    """
    perm = marlin_weight_perm_8bit()
    inv = [0] * 1024
    for j, x in enumerate(perm):
        inv[x] = j
    sp = marlin_scale_perm()
    inv_sp = [0] * 64
    for j, x in enumerate(sp):
        inv_sp[x] = j
    sw = (0, 2, 1, 3)
    smap = [(inv_sp[x] // 4) * 4 + sw[inv_sp[x] % 4] for x in range(64)]
    return (torch.tensor(inv, dtype=torch.int32), torch.tensor(smap, dtype=torch.int32))


def _tables(device):
    key = str(device)
    if key not in _deq['tables']:
        _deq['tables'][key] = tuple(t.to(device) for t in dequant_tables_cpu())
    return _deq['tables'][key]


_triton_kernel = None


def _kernel():
    global _triton_kernel
    if _triton_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def glm_d2_marlin_mxfp8_dequant(B, S, OUT, INVP, SMAP, N, K,
                                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
            k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
            c = (n // 16) * 256 + (k % 16) * 16 + (n % 16)
            p = c % 1024
            off = (k // 16) * (N * 16) + (c - p) + tl.load(INVP + p)
            q = tl.load(B + off).to(tl.int32)
            e = (q >> 3) & 15
            m = (q & 7).to(tl.float32)
            e_f = tl.where(e == 0, -6, e - 7)
            mag = tl.where(e == 0, m * 0.125, 1.0 + m * 0.125)  # E4M3: (1+m/8)*2^(e-7)
            qs = (k // 32) * N + n
            r = qs % 64
            sb = tl.load(S + (qs - r) + tl.load(SMAP + r)).to(tl.int32)
            # exact 2^(e + sb - 127) from bits (no exp2/ftz): fp32 normal range here
            ex = e_f + sb
            pw = tl.where(ex > 0, ex, 1)
            scale = (pw << 23).to(tl.float32, bitcast=True)
            val = mag * scale
            val = tl.where(ex > 0, val, 0.0)  # below fp32-normal: not produced by real weights
            val = tl.where((q >> 7) == 1, -val, val)
            tl.store(OUT + n * K + k, val.to(tl.bfloat16))

        _triton_kernel = glm_d2_marlin_mxfp8_dequant
    return _triton_kernel


_relayout_kernel = None


def _kernel_relayout():
    global _relayout_kernel
    if _relayout_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def glm_d2_marlin_to_mxfp8(B, S, WOUT, SOUT, INVP, SMAP, N, K, KBP,
                                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
            k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
            c = (n // 16) * 256 + (k % 16) * 16 + (n % 16)
            p = c % 1024
            off = (k // 16) * (N * 16) + (c - p) + tl.load(INVP + p)
            # The grid covers round_up(N, 128) rows: rows >= N are the swizzle padding.
            valid = n < N
            tl.store(WOUT + n * K + k, tl.load(B + off, mask=valid), mask=valid)
            # E8M0 scales of this tile -> 128x4-swizzled [round_up(N,128), KBP] layout;
            # padding rows get 0 so no stale scales from a bigger layer stay in the
            # shared scratch (D2d: layer-1 fused_qkv_a, N=2624 -> 2688, failed the check).
            kb = tl.program_id(1) * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)[None, :]
            qs = kb * N + n
            r = qs % 64
            sb = tl.load(S + (qs - r) + tl.load(SMAP + r), mask=valid, other=0)
            so = ((n // 128) * (KBP * 128) + (kb // 4) * 512 + (n % 32) * 16
                  + ((n % 128) // 32) * 4 + kb % 4)
            tl.store(SOUT + so, sb)

        _relayout_kernel = glm_d2_marlin_to_mxfp8
    return _relayout_kernel


def _round_up(x, m):
    return (x + m - 1) // m * m


def swizzled_scale_len(rows, k):
    return _round_up(rows, 128) * _round_up(k // 32, 4)


def swizzle_scales_128x4(s):
    """uint8 e8m0 [rows, k/32] -> flat 128x4-swizzled (FlashInfer SfLayout.layout_128x4,
    what mxfp8_quantize(is_sf_swizzled_layout=True) emits and the SM120 GEMM reads)."""
    rows, kb = s.shape
    rp, kp = _round_up(rows, 128), _round_up(kb, 4)
    pad = torch.zeros(rp, kp, dtype=s.dtype, device=s.device)
    pad[:rows, :kb] = s
    # [rp/128, 4, 32, kp/4, 4] (m_tile, inner_m, outer_m, k_tile, inner_k)
    #   -> [m_tile, k_tile, outer_m, inner_m, inner_k]
    return pad.view(rp // 128, 4, 32, kp // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def marlin_to_mxfp8(weight, weight_scale, n, k, wout, sout):
    """Marlin-packed MXFP8 -> plain FP8 [n, k] (wout) + 128x4-swizzled scales (sout)."""
    invp, smap = _tables(weight.device)
    kbp = _round_up(k // 32, 4)
    w = wout[:n * k]
    so = sout[:swizzled_scale_len(n, k)]
    grid = (_round_up(n, 128) // BN, k // BK)  # rows >= n: zero the swizzle padding
    _kernel_relayout()[grid](weight.view(torch.uint8), weight_scale.view(torch.uint8), w, so,
                             invp, smap, n, k, kbp, BLOCK_N=BN, BLOCK_K=BK)
    return w.view(torch.float8_e4m3fn).view(n, k), so


def dequant_marlin(weight, weight_scale, n, k, out):
    """Marlin-packed MXFP8 (int32 [k/16, n*4], e8m0 [k/32, n]) -> out[:n*k] as bf16 [n, k]."""
    invp, smap = _tables(weight.device)
    o = out[:n * k].view(n, k)
    _kernel()[(n // BN, k // BK)](weight.view(torch.uint8), weight_scale.view(torch.uint8), o,
                                  invp, smap, n, k, BLOCK_N=BN, BLOCK_K=BK)
    return o


def reference_dequant(q, s, rows=None):
    """Plain FP8 [n, k] + uint8 e8m0 [n, k/32] -> bf16, exact (torch)."""
    q, s = (q, s) if rows is None else (q[rows], s[rows])
    si = s.to(torch.int32)
    scale = torch.where(si == 0, torch.full_like(si, 1 << 22), si << 23).view(torch.float32)
    return (q.float().view(q.shape[0], -1, 32) * scale.unsqueeze(-1)).view(q.shape).to(torch.bfloat16)


def _scratch(numel, device):
    buf = _deq['scratch']
    if buf is None or buf.numel() < numel:
        _deq['scratch'] = None
        del buf
        _deq['scratch'] = torch.empty(numel, dtype=torch.bfloat16, device=device)
    return _deq['scratch']


def _w8_scratch(n, k, device):
    """Grow-only FP8 weight + swizzled-scale scratch (allocated at load)."""
    if _w8['w'] is None or _w8['w'].numel() < n * k:
        _w8['w'] = None
        _w8['w'] = torch.empty(n * k, dtype=torch.uint8, device=device)
    sl = swizzled_scale_len(n, k)
    if _w8['s'] is None or _w8['s'].numel() < sl:
        _w8['s'] = None
        _w8['s'] = torch.zeros(sl, dtype=torch.uint8, device=device)
    return _w8['w'], _w8['s']


def _w8_disable(msg, *args):
    _w8['ok'] = False
    log.error('GLM D2 FP8 W8A8: ' + msg + '; large-M path falls back to dequant + BF16 GEMM',
              *args)
    dev = _w8['w'].device if _w8['w'] is not None else 'cuda'
    _w8['w'] = _w8['s'] = None  # free the FP8 scratch, keep memory flat
    _scratch(_deq['max_nk'], dev)


def _w8_resources(device):
    if _w8['mod'] is None:
        from flashinfer.gemm.gemm_base import _load_gemm_sm120_mxfp8_module
        from flashinfer.quantization.fp8_quantization import mxfp8_quantize

        if device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] != 12:
            raise RuntimeError('SM120/121 MXFP8 module needs cc 12.x, got %s'
                               % (torch.cuda.get_device_capability(device),))
        _w8['mod'] = _load_gemm_sm120_mxfp8_module()
        _w8['quant'] = mxfp8_quantize
        _w8['ws'] = torch.empty(32 << 20, dtype=torch.uint8, device=device)


def w8a8_linear(x2, weight, weight_scale, n, k, tactic=None):
    """x2 [M, k] bf16 -> [M, n] bf16 via MXFP8 x MXFP8 (FlashInfer CUTLASS SM120)."""
    wq, ssw = marlin_to_mxfp8(weight, weight_scale, n, k, _w8['w'], _w8['s'])
    a_q, a_sf = _w8['quant'](x2.contiguous(), True)
    out = torch.empty(x2.shape[0], n, dtype=torch.bfloat16, device=x2.device)
    if tactic is None:
        tactic = _pick_tactic(n, k, x2.shape[0])
    _w8['mod'].mxfp8_gemm(a_q, wq, a_sf, ssw, out, _w8['ws'], tactic)
    return out


def _pick_tactic(n, k, m):
    tab = _w8['tactics'].get((n, k))
    if not tab:
        return 0
    best = tab[0][1]
    for mb, tac in tab:
        if m >= mb:
            best = tac
    return best


def _tune(layer, n, k):
    lo = max([b for b in TUNE_MS if b <= DEQUANT_M] or [TUNE_MS[0]])
    ms = [b for b in TUNE_MS if b >= lo]  # M in [b, next b) uses b's tactic; > 2048 uses 2048's
    if os.environ.get('GLM_D2_FP8_W8A8_TUNE', '1') != '1' or layer.weight.device.type != 'cuda':
        _w8['tactics'][(n, k)] = [(ms[0], 0)]
        return
    nt = int(_w8['mod'].mxfp8_gemm_tactic_num())
    x = torch.randn(ms[-1], k, dtype=torch.bfloat16, device=layer.weight.device)
    tab = []
    for m in ms:
        xm = x[:m]
        best = (float('inf'), 0)
        for tac in range(nt):
            try:
                for _ in range(2):
                    w8a8_linear(xm, layer.weight, layer.weight_scale, n, k, tac)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(3):
                    w8a8_linear(xm, layer.weight, layer.weight_scale, n, k, tac)
                b.record()
                b.synchronize()
                best = min(best, (a.elapsed_time(b) / 3, tac))
            except Exception as e:  # tactic not valid for this shape
                log.debug('GLM D2 FP8 W8A8 tactic %d [%d,%d] M=%d failed: %s', tac, n, k, m, e)
        tab.append((m, best[1]))
    _w8['tactics'][(n, k)] = tab
    del x
    log.info('GLM D2 FP8 W8A8 [%d, %d] tactics (M>=: tactic) %s of %d', n, k, tab, nt)


def _setup_w8a8(layer, prefix, q, s, n, k, ref_w) -> None:
    """Load-time: resources, bit-exact relayout check, per-shape GEMM check + tactics."""
    dev = q.device
    _w8_scratch(n, k, dev)
    if dev.type == 'meta':
        return
    try:
        _w8_resources(dev)
    except Exception as e:  # noqa: BLE001
        return _w8_disable('FlashInfer SM120 MXFP8 unavailable (%s: %s)', type(e).__name__, e)
    if os.environ.get('GLM_D2_FP8_W8A8_CHECK', '1') != '1':
        if (n, k) not in _w8['tactics']:
            _tune(layer, n, k)
        return
    wq, ssw = marlin_to_mxfp8(layer.weight, layer.weight_scale, n, k, _w8['w'], _w8['s'])
    if not torch.equal(wq.view(torch.uint8), q.view(torch.uint8)):
        return _w8_disable('relayout self-check FAILED on %s [%d, %d] (weight bytes)', prefix, n, k)
    if not torch.equal(ssw, swizzle_scales_128x4(s.view(torch.uint8))):
        return _w8_disable('relayout self-check FAILED on %s [%d, %d] (scales)', prefix, n, k)
    if (n, k) in _w8['checked']:
        return
    try:
        _tune(layer, n, k)
        g = torch.Generator(device=dev).manual_seed(n * 131 + k)
        x = torch.randn(256, k, generator=g, device=dev).to(torch.bfloat16)
        y = w8a8_linear(x, layer.weight, layer.weight_scale, n, k).float()
        ref = torch.nn.functional.linear(x, ref_w).float()
        rel = ((y - ref).norm() / ref.norm()).item()
    except Exception as e:  # noqa: BLE001
        return _w8_disable('GEMM check raised on %s [%d, %d] (%s: %s)', prefix, n, k,
                           type(e).__name__, e)
    if not rel <= W8A8_TOL:  # also catches NaN
        return _w8_disable('GEMM check FAILED on %s [%d, %d]: rel err %.4g > %.3g', prefix, n, k,
                           rel, W8A8_TOL)
    _w8['checked'].add((n, k))
    log.info('GLM D2 FP8 W8A8 check %s [%d, %d]: rel err vs BF16 %.4f (tol %.3g)', prefix, n, k,
             rel, W8A8_TOL)


def _setup_dequant(layer, prefix, q, s) -> bool:
    """Per-layer eligibility for the large-M path + bit-exact load-time checks."""
    if not DISPATCH:
        return False
    _ensure_op()
    n, k = q.shape
    if prefix.endswith('lm_head') or n % BN or k % BK or layer.weight.shape != (k // 16, n * 4):
        return False  # lm_head only ever sees sampled rows; odd/padded shapes stay Marlin
    _deq['max_nk'] = max(_deq['max_nk'], n * k)
    w8 = _w8['ok']
    # W8A8 keeps no BF16 scratch: the dequant check (and the W8A8 reference) use a temp.
    buf = (torch.empty(n * k, dtype=torch.bfloat16, device=q.device) if w8
           else _scratch(_deq['max_nk'], q.device))
    if not _deq['ok']:
        return True
    o = None
    if os.environ.get('GLM_D2_FP8_DEQUANT_CHECK', '1') == '1' and q.device.type != 'meta':
        o = dequant_marlin(layer.weight, layer.weight_scale, n, k, buf)
        for r0 in range(0, n, 1024):
            rows = slice(r0, min(n, r0 + 1024))
            if not torch.equal(o[rows], reference_dequant(q, s, rows)):
                _deq['ok'] = False
                log.error('GLM D2 FP8: dequant self-check FAILED on %s [%d, %d] rows %d+; '
                          'large-M dispatch DISABLED, all rows use Marlin', prefix, n, k, r0)
                return True
        _deq['checked'] += 1
    if w8:
        if o is None and q.device.type != 'meta':
            o = reference_dequant(q, s)
        _setup_w8a8(layer, prefix, q, s, n, k, o)
    del buf, o
    return True


def _op_impl(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
             workspace: torch.Tensor, bias: torch.Tensor | None, size_n: int,
             size_k: int, allow_dequant: bool) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
    )
    x2 = x.reshape(-1, size_k)
    if allow_dequant and _deq['ok'] and x2.shape[0] >= DEQUANT_M:
        if _w8['ok']:
            out = w8a8_linear(x2, weight, weight_scale, size_n, size_k)
            if bias is not None:
                out += bias
        else:
            w = dequant_marlin(weight, weight_scale, size_n, size_k,
                               _scratch(size_n * size_k, x.device))
            out = torch.nn.functional.linear(x2, w, bias)
    else:
        out = apply_mxfp8_marlin_linear(x2, weight, weight_scale, workspace,
                                        size_n, size_k, bias)
    return out.reshape(*x.shape[:-1], size_n)


def _op_fake(x, weight, weight_scale, workspace, bias, size_n, size_k, allow_dequant):
    return x.new_empty((*x.shape[:-1], size_n))


def _ensure_op():
    if _deq['op']:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name='glm_d2_mxfp8_linear', op_func=_op_impl,
                              fake_impl=_op_fake)
    _deq['op'] = True
    log.info('GLM D2 FP8 dispatch: rows >= %d -> %s, else Marlin', DEQUANT_M,
             'MXFP8 W8A8 GEMM (FlashInfer SM120)' if _w8['ok'] else 'dequant + BF16 GEMM')


class GlmD2Mxfp8LinearMethod(UnquantizedLinearMethod):
    """BF16 create/load (inherited); MXFP8 Marlin after loading."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def process_weights_after_loading(self, layer):
        if getattr(layer, '_glm_d2_kernel', None) is not None:
            return
        if not _convert(layer, self.prefix):
            super().process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        if getattr(layer, '_glm_d2_kernel', None) is None:
            return super().apply(layer, x, bias)
        return _apply(layer, x, bias)


class GlmD2Mxfp8HeadMethod(UnquantizedEmbeddingMethod):
    """lm_head: BF16 load (padded vocab shard), MXFP8 Marlin logits GEMM."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def process_weights_after_loading(self, layer):
        if getattr(layer, '_glm_d2_kernel', None) is not None:
            return
        if not _convert(layer, self.prefix):
            super().process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        if getattr(layer, '_glm_d2_kernel', None) is None:
            return super().apply(layer, x, bias)
        return _apply(layer, x, bias)


SIGNATURE_VERSION = 'mxfp8-marlin-v3'


def signature() -> str:
    # The threshold is read inside the op at run time, so only on/off is keyed.
    return '%s|dispatch=%d|big=%s|exclude=%s|head=%s' % (
        SIGNATURE_VERSION, int(DISPATCH), BIG if DISPATCH else '-',
        os.environ.get('GLM_D2_FP8_EXCLUDE', DEFAULT_EXCLUDE),
        os.environ.get('GLM_D2_FP8_HEAD', '1'))


def stamp_compile_key(vllm_config) -> None:
    """Put the D2 layout into vllm_config.additional_config.

    VllmConfig.compute_hash() hashes additional_config into the torch.compile /
    AOT cache key, but not the quant method's weight layout. Without this a D2
    boot loads the baseline's AOT artifact, whose graph expects BF16 [N, K]
    weights (boot1: 'expected size 384==2624' = Marlin fused_qkv_a vs BF16).
    """
    if vllm_config is None:
        return
    ac = vllm_config.additional_config
    if isinstance(ac, dict):
        ac['glm_d2_fp8'] = signature()
    else:
        raise RuntimeError('GLM D2 FP8 needs a dict additional_config to key the compile cache')


def maybe_method(layer: torch.nn.Module, prefix: str, is_lm_head: bool):
    """Hook for Exl3Config.get_quant_method on non-EXL3 BF16 linears."""
    if not eligible(prefix, is_lm_head):
        return None
    from vllm.config import get_current_vllm_config_or_none

    stamp_compile_key(get_current_vllm_config_or_none())
    if not _stats['modules'] and not _stats.get('announced'):
        _stats['announced'] = True
        log.info('GLM D2 FP8 overlay active (exclude=%s, head=%s)',
                 os.environ.get('GLM_D2_FP8_EXCLUDE', DEFAULT_EXCLUDE),
                 os.environ.get('GLM_D2_FP8_HEAD', '1'))
    if is_lm_head:
        return GlmD2Mxfp8HeadMethod(prefix)
    return GlmD2Mxfp8LinearMethod(prefix)
