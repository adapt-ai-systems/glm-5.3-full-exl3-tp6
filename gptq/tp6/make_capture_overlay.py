#!/usr/bin/env python3
"""Write overlay/d2_fp8_capture.py = the D8 overlay d2_fp8.py (sha-pinned; D8 stays OFF unless GLM_D8_W4=1) + an
activation capture for the deep-layer GPTQ-vs-RTN check. Capture boot only (eager, not for serving).

  GLM_D8_CAPTURE=<regex>     module prefixes to capture (default: layers 3/40/77 fused_qkv_a, o_proj, shared gate_up/down)
  GLM_D8_CAPTURE_DIR=<dir>   control + output dir (bind-mounted rw). Phases by marker file in that dir:
     (none)     -> calibration: accumulate H = sum x^T x (fp32, on GPU) + row count per module
     'eval'     -> held-out: store raw input rows (bf16, up to GLM_D8_CAPTURE_EVAL_ROWS=6000 per module)
     'dump'     -> write <dir>/r<rank>/<prefix>.pt {H, n, X_eval} once, then 'dumped-r<rank>' marker
Only TP rank 0 captures the GLM_D8_CAPTURE modules (H + held-out X, for deep_study.py).
  GLM_D8_CAPTURE_FULL=1      also accumulate H (no X) for EVERY D8 target module (layers 0-77 fused_qkv_a, o_proj,
                             [shared_experts.]gate_up/down) for the full GPTQ solve: row-parallel inputs (o_proj, down)
                             on every rank; replicated inputs (fused_qkv_a, gate_up) once, on rank layer % 6.
                             ~7 GB GPU/rank -> the capture boot lowers kv-cache-memory-bytes. Markers go on EVERY node. Hook point: the D2 custom op body (_op_impl), which runs in
Python on every call when the boot has --enforce-eager (no CUDA-graph replay); module = weight.data_ptr() lookup.
"""
import hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
SRC, OUT = D.parent / 'overlay/d2_fp8.py', D / 'overlay/d2_fp8_capture.py'
D8_SHA = 'ab25d026ece0216ff8c3a495dab0cceff4229f938cf4d3f8ea8fe240fdcbb4c6'
s = SRC.read_text()
assert hashlib.sha256(s.encode()).hexdigest() == D8_SHA, 'D8 overlay changed'


def replace(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


# register every converted module's final Marlin weight pointer -> prefix
s = replace(s, "    layer._glm_d2_kernel = kernel\n",
            "    layer._glm_d2_kernel = kernel\n    _cap_register(layer, prefix)\n")
# fingerprint every target's deployed BF16 weight (per-rank slice, before FP8 conversion) -> solver/loader slice check
s = replace(s, "    dtype = w.dtype\n    q, s = quantize_mxfp8(w)\n",
            "    dtype = w.dtype\n    _cap_fp(prefix, w)\n    q, s = quantize_mxfp8(w)\n")
# capture in the op body (eager boot: runs every call)
s = replace(s, "    x2 = x.reshape(-1, size_k)\n    if allow_dequant and _deq['ok'] and x2.shape[0] >= DEQUANT_M:\n        if _w8['ok']:",
            "    x2 = x.reshape(-1, size_k)\n    _cap(weight, x2)\n    if allow_dequant and _deq['ok'] and x2.shape[0] >= DEQUANT_M:\n        if _w8['ok']:")
s += r'''

# ---------------------------------------------------------------- D8-GPTQ activation capture (capture boots only)
CAP_RE = os.environ.get('GLM_D8_CAPTURE', r'model\.layers\.(3|40|77)\.(self_attn\.(fused_qkv_a_proj|o_proj)|'
                        r'mlp\.shared_experts\.(gate_up_proj|down_proj))')
CAP_DIR = os.environ.get('GLM_D8_CAPTURE_DIR', '')
CAP_EVAL_ROWS = int(os.environ.get('GLM_D8_CAPTURE_EVAL_ROWS', '6000'))
CAP_FULL = os.environ.get('GLM_D8_CAPTURE_FULL', '0') == '1'
CAP_FULL_RE = (r'model\.layers\.([0-9]|[1-6][0-9]|7[0-7])\.'
               r'(self_attn\.(fused_qkv_a_proj|o_proj)|mlp\.(shared_experts\.)?(gate_up_proj|down_proj))')
CAP_TP = int(os.environ.get('GLM_D8_CAPTURE_TP', '6'))
_capst = {'by_ptr': {}, 'mods': {}, 'phase': 'cal', 'calls': 0, 'rank': None, 'role': {}}


def _cap_rank():
    if _capst['rank'] is None:
        try:
            import torch.distributed as dist
            _capst['rank'] = dist.get_rank() if dist.is_initialized() else 0
        except Exception:  # noqa: BLE001
            _capst['rank'] = 0
    return _capst['rank']


def _cap_register(layer, prefix):
    if CAP_DIR and (re.fullmatch(CAP_RE, prefix) or (CAP_FULL and re.fullmatch(CAP_FULL_RE, prefix))):
        _capst['by_ptr'][layer.weight.data_ptr()] = prefix
        log.info('GLM D8 capture: registered %s', prefix)


def _cap_phase():
    _capst['calls'] += 1
    if _capst['phase'] == 'dumped':
        return 'dumped'
    d = Path(CAP_DIR)
    ph = 'dump' if (d / 'dump').exists() else 'eval' if (d / 'eval').exists() else 'cal'
    if ph == 'dump' and _capst['phase'] != 'dumped':
        _cap_dump()
        ph = 'dumped'
    elif _capst['phase'] == 'dumped':
        ph = 'dumped'
    _capst['phase'] = ph
    return ph


def _cap(weight, x2):
    if not CAP_DIR or not _capst['by_ptr']:
        return
    prefix = _capst['by_ptr'].get(weight.data_ptr())
    role = _cap_role(prefix) if prefix is not None else (False, False)
    ph = _cap_phase()
    if not role[0] or ph not in ('cal', 'eval'):
        return
    m = _capst['mods'].setdefault(prefix, {'H': None, 'n': 0, 'X': []})
    if ph == 'cal':
        xf = x2.float()
        h = xf.t() @ xf
        m['H'] = h if m['H'] is None else m['H'].add_(h)
        m['n'] += xf.shape[0]
    elif ph == 'eval' and role[1]:
        have = sum(t.shape[0] for t in m['X'])
        if have < CAP_EVAL_ROWS:
            m['X'].append(x2[:CAP_EVAL_ROWS - have].detach().to(torch.bfloat16).cpu())


def fingerprint(w):
    """[n, k] -> fp32 [n] = w @ v, v a fixed integer-hash vector in [-0.5, 0.5) (identical on CPU and GPU)."""
    k = w.shape[1]
    v = ((torch.arange(k, dtype=torch.int64, device=w.device) * 2654435761) % 1009).float() / 1009 - 0.5
    return w.float() @ v


def _cap_fp(prefix, w):
    if CAP_DIR and (re.fullmatch(CAP_RE, prefix) or (CAP_FULL and re.fullmatch(CAP_FULL_RE, prefix))):
        _capst.setdefault('fp', {})[prefix] = (fingerprint(w).cpu(), tuple(w.shape))


def _cap_role(prefix):
    """(capture H on this rank, also keep held-out X rows)"""
    r = _capst['role'].get(prefix)
    if r is None:
        rank = _cap_rank()
        deep = rank == 0 and re.fullmatch(CAP_RE, prefix) is not None
        full = False
        if CAP_FULL and re.fullmatch(CAP_FULL_RE, prefix):
            if prefix.endswith(('o_proj', 'down_proj')):
                full = True
            else:
                full = int(prefix.split('.')[2]) % CAP_TP == rank
        r = _capst['role'][prefix] = (deep or full, deep)
    return r


def _cap_dump():
    d = Path(CAP_DIR) / ('r%d' % _cap_rank())
    d.mkdir(parents=True, exist_ok=True)
    for prefix in list(_capst['mods']):
        m = _capst['mods'].pop(prefix)  # free each H after saving (unified memory: .cpu() is a second copy)
        torch.save({'H': None if m['H'] is None else m['H'].cpu(), 'n': m['n'],
                    'X': torch.cat(m['X']) if m['X'] else None}, d / (prefix + '.pt'))
        log.info('GLM D8 capture: dumped %s n=%d eval_rows=%d', prefix, m['n'], sum(t.shape[0] for t in m['X']))
        del m
    torch.save(_capst.get('fp', {}), d / 'fingerprints.pt')
    (Path(CAP_DIR) / ('dumped-r%d' % _cap_rank())).touch()
'''
s = replace(s, "import logging\nimport os\nimport re\n", "import logging\nimport os\nimport re\nfrom pathlib import Path\n")
compile(s, str(OUT), 'exec')
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(s)
print('wrote', OUT, hashlib.sha256(s.encode()).hexdigest())
