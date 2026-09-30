import sys, json, os; sys.path.insert(0, "/work")
import torch, bench_layer as B
from torch.profiler import profile, ProfilerActivity
m, layer, ref, info = B.load_layer()
r = B.Router(); g = torch.Generator().manual_seed(5)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=B.DEV)
out = {}
NIT = 10
for M in (5, 20):
    x = B.make_x(M, g); w, ids = r(x)
    for _ in range(3): m._apply_mixed_rank_sliced(layer, x, w, ids)
    gr = torch.cuda.CUDAGraph()
    xs, ws, is_ = x.clone(), w.clone(), ids.clone()
    with torch.cuda.graph(gr): m._apply_mixed_rank_sliced(layer, xs, ws, is_)
    for mode in ("eager", "graph"):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(NIT):
                flush.zero_()
                torch.cuda.synchronize()
                with torch.profiler.record_function("MOE_CALL"):
                    if mode == "eager": m._apply_mixed_rank_sliced(layer, x, w, ids)
                    else: gr.replay()
                torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernels = sorted([(e.time_range.start, e.time_range.end - e.time_range.start, e.name) for e in evs])
        # split into calls by the flush kernel (elementwise fill) -> keep kernels after each flush
        calls, cur = [], None
        for s, d, n in kernels:
            if "fill" in n.lower() or "zero" in n.lower() and d > 20 and "MemsetD" in n or n.startswith("Memset"):
                cur = []; calls.append(cur); continue
            if cur is not None: cur.append((s, d, n))
        agg = {}
        for c in calls:
            for s, d, n in c: agg.setdefault(n, []).append(d)
        summary = [dict(kernel=n[:110], calls_per_moe=len(v) / len(calls), avg_us_each=sum(v) / len(v), us_per_moe=sum(v) / len(calls))
                   for n, v in agg.items()]
        summary.sort(key=lambda r: -r["us_per_moe"])
        mid = calls[len(calls) // 2]
        t0 = mid[0][0]
        timeline = [dict(t_start_us=s - t0, dur_us=d, kernel=n[:110]) for s, d, n in mid]
        span = (mid[-1][0] + mid[-1][1] - t0) if mid else 0
        out[f"M{M}_{mode}"] = dict(n_calls=len(calls), span_us_first_to_last_kernel=span, sum_kernel_us=sum(d for _, d, _ in mid),
                                   summary=summary, timeline=timeline)
        print(f"== M={M} {mode}: {len(calls)} calls, span {span:.1f} us, sum kernels {sum(d for _,d,_ in mid):.1f} us")
        for row in summary[:12]: print("  %8.1f us/moe  x%.1f  avg %8.1f  %s" % (row["us_per_moe"], row["calls_per_moe"], row["avg_us_each"], row["kernel"]))
        for row in timeline: print("    t=%8.1f dur=%8.1f %s" % (row["t_start_us"], row["dur_us"], row["kernel"][:90]))
json.dump(out, open("/work/out/profile.json", "w"), indent=1)
