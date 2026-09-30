"""Freed-head TP4 real-layer probe. Import/--help are CPU-only; main uses CUDA.

One native prepared layer, M65/256/2048, warm changing-input graph and two
stream-owned arenas. No full model, TP6 export, cache build, restore or soak.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace as NS


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='/model')
    ap.add_argument('--rank', type=int, choices=range(4), default=0)
    ap.add_argument('--layer', type=int, default=3)
    ap.add_argument('--capacity', type=int, choices=[2048], default=2048)
    ap.add_argument('--rows', nargs='+', type=int, choices=[65, 256, 2048], default=[65, 256, 2048])
    ap.add_argument('--output', type=Path, required=True)
    return ap


def load_layer(model, rank, index, capacity):
    import torch
    from transformers import PretrainedConfig
    from vllm.model_executor.layers.quantization import exl3
    import tr3_runtime
    from .hook import install, validate_layer

    raw = json.loads((Path(model) / 'config.json').read_text())
    config = exl3.Exl3Config.from_config(raw['quantization_config'])
    config.maybe_update_config(model, PretrainedConfig.from_dict(raw))
    name = f'model.layers.{index}.mlp.experts'
    layer = NS(layer_name=name, local_num_experts=256,
               exl3_layer_bitrates=tuple(config.rank_sliced_layer_bitrates(name)),
               exl3_hidden_size=6144, exl3_intermediate_size_per_partition=512,
               exl3_tp_rank=rank, exl3_rank_sliced=True, exl3_mixed_bitrate=True,
               exl3_max_num_batched_tokens=capacity, activation=NS(value='silu'))
    for group in ('w13', 'w2'):
        for field in ('suh', 'svh', 'trellis', 'mcg', 'mul1'):
            setattr(layer, f'{group}_{field}', NS(device=torch.device('cuda:0')))
    # attach retains normal identity/schema/shape/checksum verification. On a
    # miss stop this probe; do not build a cache or load a second checkpoint.
    tr3_runtime.settings.cache_clear()
    tr3_runtime.attach(layer)
    if getattr(layer, '_tr3_cache_entry', None) is None:
        raise RuntimeError('Native prepared layer not validated; no slow-loader fallback in fixture')
    method = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
    method.quant_config = config
    if not tr3_runtime.restore(method, layer, exl3._load_b12x_mixed_trellis()):
        raise RuntimeError('Native restore failed')
    validate_layer(layer)
    mixed = layer.exl3_mixed_trellis
    order = [e for ids in mixed['tier_ids'] for e in ids]
    expected_mapping = [order.index(e) for e in range(256)]
    if mixed['global_to_combined'].cpu().tolist() != expected_mapping:
        raise RuntimeError('Native TP4 mapping does not match tier pointer order')
    install(exl3)
    native = exl3.Exl3MoEMethod._apply_mixed_rank_sliced.__wrapped__
    return layer, lambda x, w, ids: native(method, layer, x, w, ids), method._apply_mixed_rank_sliced


def error(actual, expected):
    import torch
    a, b = actual.float(), expected.float()
    return dict(finite=bool(torch.isfinite(a).all()),
                relative_rms=float(((a-b).square().mean()/b.square().mean()).sqrt()),
                cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                max_abs=float((a-b).abs().max()))


def passes(row):
    # Same exploratory thresholds as the TP6 comparator, not qualification.
    return row['finite'] and row['relative_rms'] < .01 and row['cosine'] > .99995


def timed(fn):
    import torch
    for _ in range(2):
        fn()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(5):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/5


def run(args, result, save):
    import torch
    from . import runtime, serving
    os.environ.update(TR3_LOAD_ACCEL='1', TR3_CACHE_MODEL=args.model,
                      TP4_E3_PREFILL='1', GLM6_E3_AUDIT='0')
    torch.manual_seed(2026091303)
    save('LOAD_BEGIN')
    layer, reference, wrapped = load_layer(args.model, args.rank, args.layer, args.capacity)
    candidate = lambda x, w, ids: wrapped(layer, x, w, ids)
    save('LAYER_READY', experts=256, tiers=[[3, 192], [4, 64]])
    for m in args.rows:
        x = torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
        ids = (torch.arange(m*8, device='cuda').reshape(m, 8) % 256).contiguous()
        weights = torch.rand(m, 8, device='cuda', dtype=torch.float32)
        weights /= weights.sum(1, keepdim=True)
        routes = runtime.route_tables(ids, weights, layer.exl3_mixed_trellis['global_to_combined'], 256)
        route_stats = dict(actual_rows=int(routes['num_rows'].item()),
                           actual_segments=int(routes['num_segs'].item()),
                           table_segments=routes['seg_expert'].numel())
        assert route_stats['actual_rows'] == m*8
        assert route_stats['actual_segments'] <= route_stats['table_segments'] <= 512
        del routes
        expected = reference(x, weights, ids)
        actual = candidate(x, weights, ids)
        torch.cuda.synchronize()
        row = dict(rows=m, **route_stats, **error(actual, expected))
        result['rows'].append(row)
        save('NUMERICS', **row)
        if not passes(row):
            raise RuntimeError('E3 exploratory numerical gate failed')
        row.update(b12x_layer_ms=timed(lambda: reference(x, weights, ids)),
                   e3_layer_ms=timed(lambda: candidate(x, weights, ids)))
        row['layer_speedup'] = row['b12x_layer_ms']/row['e3_layer_ms']
        save('TIMING', **row)

    # Two side streams; warm the same stream used for graph capture. Capture
    # on an unprepared stream is separately CPU-tested to choose native B12X.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            candidate(x, weights, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=streams[0]):
        if not serving.can_apply(layer, x, ids):
            raise RuntimeError('Warm graph would exercise B12X rather than E3')
        graph_out = candidate(x, weights, ids)
    for iteration in range(2):
        x.mul_(-.98)
        ids.copy_(ids.roll(1, dims=1))
        weights.copy_(weights.roll(1, dims=1))
        graph.replay()
        torch.cuda.synchronize()
        row = dict(kind='warm_graph_changing_inputs', iteration=iteration,
                   **error(graph_out, reference(x, weights, ids)))
        result['contracts'].append(row)
        save('CONTRACT', **row)
        if not passes(row):
            raise RuntimeError('Warm graph numerical gate failed')
    other = -x
    expected = [reference(x, weights, ids), reference(other, weights, ids)]
    torch.cuda.synchronize()
    pending = []
    for j in range(8):
        with torch.cuda.stream(streams[j % 2]):
            pending.append((candidate(x if j % 2 == 0 else other, weights, ids), expected[j % 2]))
    torch.cuda.synchronize()
    for j, (actual, ref) in enumerate(pending):
        row = dict(kind='two_stream_owned', iteration=j, **error(actual, ref))
        result['contracts'].append(row)
        save('CONTRACT', **row)
        if not passes(row):
            raise RuntimeError('Two-stream numerical gate failed')
    result['arenas'] = [dict(key=list(key), bytes=sum(t.numel()*t.element_size() for t in arena.values()))
                        for key, arena in runtime._SCRATCH.items()]
    result['peak_torch_bytes'] = torch.cuda.max_memory_allocated()
    result['all_rows_faster'] = all(row['layer_speedup'] > 1 for row in result['rows'])
    result['status'] = 'TP4_E3_LAYER_PASS'


def main():
    args = parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Claim an unused evidence path before touching CUDA.
    with args.output.open('x'):
        pass
    result = dict(status='RUNNING', started_epoch=time.time(), rows=[], contracts=[],
                  layer=args.layer, rank=args.rank, capacity=args.capacity,
                  scope='Real TP4 prepared weights; synthetic routes; not full-model qualification')

    def save(event, **data):
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(event=event, epoch=time.time(), **data)), flush=True)

    try:
        run(args, result, save)
    except BaseException as exc:
        result.update(status='TP4_E3_LAYER_FAILED', error=repr(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.time()-result['started_epoch']
        save('DONE', status=result['status'])


if __name__ == '__main__':
    main()
