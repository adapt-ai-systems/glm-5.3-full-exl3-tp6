"""Direct exact-fragment tier loader; no second weight cache on disk.

Read the verified, immutable rank export into final tier-ordered CPU buffers,
then copy each contiguous buffer to the GPU once. Avoid loading thousands of
individual GPU tensors only to validate/re-stack/free them at every restart.
The original loader remains available with GLM6_FAST_LOAD=0.
"""
from __future__ import annotations
import json, logging, mmap, os, re, struct, time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from fragments import PATTERN, placement

log = logging.getLogger('vllm.glm6_fastload')
DTYPES = {'I16': '<i2', 'F16': '<f2', 'I32': '<i4'}
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


def fingerprint(path):
    s = path.stat()
    return s.st_ino, s.st_size, s.st_mtime_ns


@lru_cache(None)
def index_layer(model: str, rank: int, layer: int):
    root = Path(model)
    layout = json.loads((root / 'glm6_layout.json').read_text())
    if layout['schema'] != 'exact-fragments-v1' or layout['rank'] != rank:
        raise ValueError('Fast loader requires the matching exact-fragment rank export')
    if not (root / 'VERIFIED').is_file():
        raise ValueError('Rank export has not passed checksum verification')
    path = root / f'model-layer-{layer:03d}.safetensors'
    state = fingerprint(path)
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if n > 256 << 20:
            raise ValueError('Oversized safetensors header')
        header = json.loads(f.read(n))
    pairs = placement(rank, layer)
    local = {pair: i for i, pair in enumerate(pairs)}
    entries = {}
    for name, meta in header.items():
        m = PATTERN.fullmatch(name)
        if not m:
            continue
        if int(m[2]) != layer or (int(m[3]), int(m[5])) not in local:
            raise ValueError(f'Wrong expert ownership: {name}')
        key = (local[int(m[3]), int(m[5])], m[4], m[6])
        if key in entries:
            raise ValueError(f'Duplicate tensor: {name}')
        dtype = DTYPES[meta['dtype']]
        shape = tuple(meta['shape'])
        a, b = meta['data_offsets']
        if any(d < 0 for d in shape) or a < 0 or b < a or 8 + n + b > state[1]:
            raise ValueError(f'Invalid tensor bounds: {name}')
        if int(np.prod(shape)) * np.dtype(dtype).itemsize != b - a:
            raise ValueError(f'Invalid tensor byte count: {name}')
        entries[key] = (8 + n + a, dtype, shape)
    expected = {(e, p, f) for e in range(len(pairs)) for p in PROJECTIONS
                for f in ('trellis', 'suh', 'svh', 'mcg')}
    if entries.keys() != expected:
        raise ValueError('Incomplete expert layer in rank export')
    return path, state, entries


def assemble(entry, bits, pairs, hidden, intermediate):
    """CPU-only byte-preserving layout; small synthetic tests use this too."""
    path, state, entries = entry
    if fingerprint(path) != state:
        raise RuntimeError('Rank payload changed after indexing')
    bits = tuple(int(k) for k in bits)
    if len(bits) != len(pairs) or sorted(set(bits)) != [3, 4]:
        raise ValueError('Expected matching K3/K4 expert fragments')
    if hidden % 16 or intermediate % 16:
        raise ValueError('Invalid packed dimensions')
    tiers = [(k, tuple(e for e, b in enumerate(bits) if b == k)) for k in (3, 4)]
    count = len(bits)
    arrays = dict(gate_suh=np.empty((count, hidden), '<f2'),
                  up_suh=np.empty((count, hidden), '<f2'),
                  down_svh=np.empty((count, hidden), '<f2'),
                  intermediate=np.empty((count, 3 * intermediate), '<f2'),
                  global_to_combined=np.full(256, -1, '<i4'),
                  descriptor_map=np.empty(3 * count, '<i4'))
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        def src(expert, projection, field, shape):
            offset, dtype, actual = entries[expert, projection, field]
            wanted = '<i4' if field == 'mcg' else '<i2' if field == 'trellis' else '<f2'
            if dtype != wanted or actual != tuple(shape):
                raise ValueError(f'Source shape/dtype mismatch: {expert}/{projection}/{field}')
            return np.ndarray(actual, dtype=dtype, buffer=data, offset=offset)
        combined = 0
        for tier, (k, ids) in enumerate(tiers):
            arrays[f'w13_k{k}'] = np.empty((2, len(ids), hidden//16, intermediate//16, 16*k), '<i2')
            arrays[f'w2_k{k}'] = np.empty((len(ids), intermediate//16, hidden//16, 16*k), '<i2')
            for local, expert in enumerate(ids):
                for projection in PROJECTIONS:
                    shape = entries[expert, projection, 'mcg'][2]
                    if shape not in ((), (1,)):
                        raise ValueError('Invalid scalar marker shape')
                    if int(src(expert, projection, 'mcg', shape).reshape(-1)[0]) & 0xffffffff != 0xCBAC1FED:
                        raise ValueError('Invalid MCG codebook marker')
                    h, i = (intermediate, hidden) if projection == 'down_proj' else (hidden, intermediate)
                    if projection == 'down_proj':
                        arrays[f'w2_k{k}'][local] = src(expert, projection, 'trellis', (h//16, i//16, 16*k))
                    else:
                        arrays[f'w13_k{k}'][0 if projection == 'gate_proj' else 1, local] = src(expert, projection, 'trellis', (h//16, i//16, 16*k))
                arrays['gate_suh'][combined] = src(expert, 'gate_proj', 'suh', (hidden,))
                arrays['up_suh'][combined] = src(expert, 'up_proj', 'suh', (hidden,))
                arrays['down_svh'][combined] = src(expert, 'down_proj', 'svh', (hidden,))
                for block, (projection, field) in enumerate((('gate_proj', 'svh'), ('up_proj', 'svh'), ('down_proj', 'suh'))):
                    arrays['intermediate'][combined, block*intermediate:(block+1)*intermediate] = src(expert, projection, field, (intermediate,))
                logical_expert = pairs[expert][0]
                if arrays['global_to_combined'][logical_expert] != -1:
                    raise ValueError('Duplicate logical expert ownership')
                arrays['global_to_combined'][logical_expert] = combined
                arrays['descriptor_map'][combined::count] = (tier << 8) | local
                combined += 1
    if fingerprint(path) != state:
        raise RuntimeError('Rank payload changed while assembling')
    return arrays, tiers


def attach(layer):
    if os.environ.get('GLM6_FAST_LOAD', '0') != '1':
        return
    if not (getattr(layer, 'exl3_rank_sliced', False) and getattr(layer, 'exl3_mixed_bitrate', False)):
        raise ValueError('Fast loader only supports exact-fragment mixed-rank layers')
    index = int(re.search(r'layers\.(\d+)\.', str(layer.layer_name))[1])
    entry = index_layer(os.environ.get('GLM6_MODEL', '/model'), int(layer.exl3_tp_rank), index)
    layer._glm6_fastload_entry = entry
    for prefix in ('w13', 'w2'):
        for suffix in ('suh', 'svh', 'trellis', 'mcg', 'mul1'):
            getattr(layer, f'{prefix}_{suffix}')._tr3_cache_skip = True


def restore(method, layer, mixed_api):
    entry = getattr(layer, '_glm6_fastload_entry', None)
    if entry is None:
        return False
    import torch
    started = time.monotonic()
    hidden = int(layer.exl3_hidden_size)
    intermediate = int(layer.exl3_intermediate_size_per_partition)
    arrays, tiers = assemble(entry, layer.exl3_layer_bitrates, layer.glm6_pairs, hidden, intermediate)
    assembled = time.monotonic()
    device = layer.w13_trellis.device
    # Owning CUDA allocations, not live views onto writable source files.
    for name in list(arrays):
        arrays[name] = torch.from_numpy(arrays[name]).to(device=device, copy=True)
    copied = time.monotonic()
    configs = (method._mixed_trellis_tile_config(hidden, intermediate),
               method._mixed_trellis_prefill_tile_config(hidden, intermediate))
    prepared = [[], []]
    offset = 0
    for k, ids in tiers:
        sl = slice(offset, offset + len(ids))
        w13, w2 = arrays[f'w13_k{k}'], arrays[f'w2_k{k}']
        for objects, config in zip(prepared, configs):
            objects.append(mixed_api.prepare_weights(w13=w13, w2=w2, hidden_size=hidden,
                intermediate_size=intermediate, num_experts=len(ids), activation=layer.activation.value,
                fc1_tile_n=config[1], fc2_tile_n=config[3], params_dtype=torch.float16,
                w13_layout='trellis_t256_proj', trellis_bits=k, codebook='mcg',
                gate_suh=arrays['gate_suh'][sl], up_suh=arrays['up_suh'][sl],
                intermediate_rotations=arrays['intermediate'][sl], down_svh=arrays['down_svh'][sl],
                tile_config=config, workspace=w13.view(torch.int32).reshape(-1)[:1]))
        offset += len(ids)
    counts = tuple(len(ids) for _, ids in tiers)
    arrays['descriptor_map']._mt_projection_counts = (counts, counts)
    layer.exl3_mixed_trellis = dict(tiers=tuple(prepared[0]), prefill_tiers=tuple(prepared[1]),
        tier_ids=tuple(ids for _, ids in tiers), tier_bits=tuple(k for k, _ in tiers),
        trellis_codebook='mcg', global_to_combined=arrays['global_to_combined'],
        descriptor_map=arrays['descriptor_map'], rotations=SimpleNamespace(
            intermediate=arrays['intermediate'], gate_suh=arrays['gate_suh'],
            up_suh=arrays['up_suh'], down_svh=arrays['down_svh']),
        broadcast_suh=False, broadcast_svh=False, tile_config=configs[0], prefill_tile_config=configs[1])
    layer.exl3_trellis_tile_config = configs[0]
    log.info('GLM6 direct-tier load %s cpu=%.3fs copy=%.3fs prepare=%.3fs total=%.3fs',
             layer.layer_name, assembled-started, copied-assembled, time.monotonic()-copied, time.monotonic()-started)
    return True
