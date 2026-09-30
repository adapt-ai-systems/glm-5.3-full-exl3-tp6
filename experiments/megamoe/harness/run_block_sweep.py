import sys, os, json, statistics; sys.path.insert(0, "/work")
BM = sys.argv[1]
os.environ["VLLM_EXL3_PREFILL_BLOCK_M"] = BM
import torch, bench_layer as B
from torch.profiler import profile, ProfilerActivity
rows = []
try:
    m, layer, ref, info = B.load_layer()
    r = B.Router(); g = torch.Generator().manual_seed(21)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=B.DEV)
    for M in (1024, 2048, 4096):
        ss = []
        for _ in range(6):
            x = B.make_x(M, g); w, ids = r(x); ss.append((x, w, ids))
        row = dict(block_m=int(BM), M=M)
        try:
            out = m._apply_mixed_rank_sliced(layer, *ss[0]); torch.cuda.synchronize()
            m._apply_mixed_rank_sliced(layer, *ss[1])
            ro = B.ref_moe(ref, ss[0][0], ss[0][1], ss[0][2]); d = out.float() - ro
            row.update(rel_l2=(d.norm() / ro.norm()).item(), max_abs_err=d.abs().max().item(), ref_absmax=ro.abs().max().item(),
                       policy_block_m=layer.exl3_mixed_trellis["runtime_policy"]["prefill_block_m"])
            ts = []
            for it in range(12):
                x, w, ids = ss[it % 6]; flush.zero_()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); m._apply_mixed_rank_sliced(layer, x, w, ids); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
            ts.sort(); row.update(us_median=statistics.median(ts), us_min=ts[0], us_p90=ts[10])
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for it in range(4):
                    x, w, ids = ss[it]; flush.zero_(); m._apply_mixed_rank_sliced(layer, x, w, ids)
                torch.cuda.synchronize()
            agg = {}
            for ev in prof.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA and "MixedTrellisKernel" in ev.name:
                    agg.setdefault("main", []).append(ev.time_range.end - ev.time_range.start)
                elif ev.device_type == torch.autograd.DeviceType.CUDA and ("TopKSum" in ev.name or "pack_topk" in ev.name):
                    agg.setdefault("route+topk", []).append(ev.time_range.end - ev.time_range.start)
            row["main_grid_us"] = statistics.mean(agg["main"]); row["route_topk_us_total"] = sum(agg.get("route+topk", [0])) / 4
        except Exception as ex:
            row["error"] = repr(ex)[:600]
        print(json.dumps(row), flush=True); rows.append(row)
except Exception as ex:
    rows.append(dict(block_m=int(BM), error="setup: " + repr(ex)[:800])); print(rows[-1], flush=True)
open(f"/work/out/blocksweep_{BM}.jsonl", "w").write("\n".join(json.dumps(x) for x in rows) + "\n")
