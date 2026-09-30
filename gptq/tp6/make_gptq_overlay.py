#!/usr/bin/env python3
"""Write overlay/d2_fp8_gptq.py = the D8 overlay d2_fp8.py (sha-pinned) with pre-solved GPTQ codes/scales in place of
the load-time RTN quantizer. Everything else (kernel, op, dispatch, FP8 copy for prefill, kv_a FP8 split, load-time
kernel check) is the D8 overlay unchanged.

  GLM_D8_GPTQ_DIR=<dir>   this rank's solve_gptq.py output (layer-LLL.safetensors; <prefix>.q4/.s4/.fp, metadata
                          group, rank). Unset -> the D8 RTN path, byte-identical behaviour.
Per module (hard fail = RuntimeError at load, the boot dies loudly; never a silent RTN fallback):
  - file / tensors present, metadata rank == this TP rank, group == GLM_D8_W4_GROUP, shapes [n4, k/2] / [n4, k/g]
  - fingerprint of the live BF16 rows w[:n4] matches the stored one (rel <= GLM_D8_GPTQ_FP_TOL, default 1e-3)
  - dequant vs BF16 weight rel err <= GLM_D8_GPTQ_MAXW (default 0.2; GPTQ g64 is ~0.10-0.12, garbage is ~1)
Compile key: d8 signature becomes w4s-g<G>-gptq-v1.
"""
import hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
SRC, OUT = D.parent / 'overlay/d2_fp8.py', D / 'overlay/d2_fp8_gptq.py'
D8_SHA = 'ab25d026ece0216ff8c3a495dab0cceff4229f938cf4d3f8ea8fe240fdcbb4c6'
s = SRC.read_text()
assert hashlib.sha256(s.encode()).hexdigest() == D8_SHA, 'D8 overlay changed'


def replace(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


s = replace(s, "    return '|d8=w4s-g%d-clip-v1|inc=%s' % (D8_GROUP, D8_INCLUDE)\n",
            "    return '|d8=w4s-g%d-%s-v1|inc=%s' % (D8_GROUP, 'gptq' if D8_GPTQ_DIR else 'clip', D8_INCLUDE)\n")
s = replace(s, "    q, sc = quantize_int4_clip(w[:n4], D8_GROUP)\n",
            "    q, sc = _d8_gptq_load(prefix, w[:n4]) if D8_GPTQ_DIR else quantize_int4_clip(w[:n4], D8_GROUP)\n")
s = replace(s, "        _d8['failed'] += 1\n",
            "        _d8['failed'] += 1\n"
            "        if D8_GPTQ_DIR:\n"
            "            raise RuntimeError('GLM D8 GPTQ: load check failed on %s: kernel rel %.4g, kv_a rel %.4g' % (prefix, rel4, relkv))\n")
s = replace(s, "    del q, sc, ws, ref4, y\n",
            "    del q, sc, ws, ref4, y\n"
            "    if D8_GPTQ_DIR and not relw <= D8_GPTQ_MAXW:\n"
            "        raise RuntimeError('GLM D8 GPTQ: %s dequant vs BF16 rel err %.4g > %.3g' % (prefix, relw, D8_GPTQ_MAXW))\n")
s = replace(s, "_d8 = {'op': False, 'modules': 0, 'w4_bytes': 0, 'kv_bytes': 0, 'failed': 0}\n",
            "_d8 = {'op': False, 'modules': 0, 'w4_bytes': 0, 'kv_bytes': 0, 'failed': 0}\n"
            "D8_GPTQ_DIR = os.environ.get('GLM_D8_GPTQ_DIR', '')\n"
            "D8_GPTQ_FP_TOL = float(os.environ.get('GLM_D8_GPTQ_FP_TOL', '1e-3'))\n"
            "D8_GPTQ_MAXW = float(os.environ.get('GLM_D8_GPTQ_MAXW', '0.2'))\n")
s += r'''

# ---------------------------------------------------------------- D8-GPTQ: pre-solved codes/scales (solve_gptq.py)
def d8_fingerprint(w):
    """fp32 [n] = w @ v, v a fixed integer hash in [-0.5, 0.5); identical to solve_gptq.fingerprint."""
    k = w.shape[1]
    v = ((torch.arange(k, dtype=torch.int64, device=w.device) * 2654435761) % 1009).float() / 1009 - 0.5
    return w.float() @ v


def _d8_tp_rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001  (CPU tests without a TP group)
        return int(os.environ.get('GLM_D8_GPTQ_RANK', '0'))


def _d8_gptq_load(prefix, w):
    """-> (codes int32 [n4, k] in 0..15, bf16 scales [n4, k/G]) on w.device, after the slice/rank/format checks."""
    from safetensors import safe_open
    layer = int(prefix.split('.')[2])
    path = os.path.join(D8_GPTQ_DIR, 'layer-%03d.safetensors' % layer)
    n4, k = w.shape
    with safe_open(path, 'pt') as f:
        meta = f.metadata() or {}
        if int(meta.get('rank', -1)) != _d8_tp_rank() or int(meta.get('group', -1)) != D8_GROUP:
            raise RuntimeError('GLM D8 GPTQ: %s: file rank %s group %s, want rank %d group %d'
                               % (path, meta.get('rank'), meta.get('group'), _d8_tp_rank(), D8_GROUP))
        p4, s4, fp = (f.get_tensor(prefix + x) for x in ('.q4', '.s4', '.fp'))
    if tuple(p4.shape) != (n4, k // 2) or tuple(s4.shape) != (n4, k // D8_GROUP) or tuple(fp.shape) != (n4,):
        raise RuntimeError('GLM D8 GPTQ: %s shapes q4 %s s4 %s fp %s for weight [%d, %d]'
                           % (prefix, tuple(p4.shape), tuple(s4.shape), tuple(fp.shape), n4, k))
    fp = fp.to(w.device)
    rel = ((d8_fingerprint(w) - fp).norm() / fp.norm().clamp(min=1e-12)).item()
    if not rel <= D8_GPTQ_FP_TOL:
        raise RuntimeError('GLM D8 GPTQ: %s fingerprint mismatch rel %.3g > %.3g (wrong slice/rank/checkpoint)'
                           % (prefix, rel, D8_GPTQ_FP_TOL))
    p4 = p4.to(w.device)
    q = torch.stack([p4 & 15, p4 >> 4], -1).view(n4, k).to(torch.int32)
    if _d8['modules'] < 8:
        log.info('GLM D8 GPTQ: %s from %s (fingerprint rel %.2e)', prefix, path, rel)
    return q, s4.to(device=w.device, dtype=torch.bfloat16)
'''
compile(s, str(OUT), 'exec')
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(s)
print('wrote', OUT, hashlib.sha256(s.encode()).hexdigest())
