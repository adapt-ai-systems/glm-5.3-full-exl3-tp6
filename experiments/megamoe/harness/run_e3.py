"""Run Mia-derived E3 prefill (live TP4_E3_PREFILL path) on the same prepared layer as bench_layer."""
import sys, os, json, statistics; sys.path.insert(0, "/work"); sys.path.insert(0, "/work/e3pkg")
import torch, bench_layer as B
from torch.profiler import profile, ProfilerActivity
from e3 import runtime as e3rt
m, layer, ref, info = B.load_layer()
r = B.Router(); g = torch.Generator().manual_seed(31)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=B.DEV)
e3rt.bind(layer)
rows = []
def e3(x, w, ids): return e3rt.apply(layer, x, w, ids, stream_scratch=True)
for M in (64, 1024, 2048, 4096):
    ss = []
    for _ in range(6):
        x = B.make_x(M, g); w, ids = r(x); ss.append((x, w, ids))
    out = e3(*ss[0]); torch.cuda.synchronize(); e3(*ss[1])
    ro = B.ref_moe(ref, *ss[0]); d = out.float() - ro
    b12 = m._apply_mixed_rank_sliced(layer, *ss[0]) if M > 32 else None
    row = dict(path="E3", M=M, rel_l2=(d.norm() / ro.norm()).item(), max_abs_err=d.abs().max().item(), ref_absmax=ro.abs().max().item())
    if b12 is not None: row["rel_l2_vs_b12x"] = ((out.float() - b12.float()).norm() / b12.float().norm()).item()
    ts = []
    for it in range(12):
        x, w, ids = ss[it % 6]; flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); e3(x, w, ids); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
    ts.sort(); row.update(us_median=statistics.median(ts), us_min=ts[0], us_p90=ts[10])
    NIT = 6
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for it in range(NIT):
            x, w, ids = ss[it]; flush.zero_(); torch.cuda.synchronize()
            e3(x, w, ids); torch.cuda.synchronize()
    agg = {}
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA: continue
        n = ev.name
        if "glm6_e3" in n: k = n.split("glm6_e3_")[1].split("(")[0]
        elif "fill" in n.lower() or "Memset" in n:
            k = "zero_or_flush(fill/memset)"
        else: k = "glue:" + n[:60]
        agg.setdefault(k, []).append(ev.time_range.end - ev.time_range.start)
    # flush is one big fill per iter (256MB) -> subtract by dropping the largest fill entries of duration>... report raw and note
    row["kernels_us_per_call"] = {k: sum(v) / NIT for k, v in agg.items()}
    row["kernel_counts_per_call"] = {k: len(v) / NIT for k, v in agg.items()}
    print(json.dumps(row), flush=True); rows.append(row)
open("/work/out/e3.jsonl", "w").write("\n".join(json.dumps(x) for x in rows) + "\n")
