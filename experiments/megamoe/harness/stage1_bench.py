"""Stage 1 harness: E3 gather+gateup -> {E3 down (reference), exl3mm fc2+finalize}. Correctness + timing."""
import sys, os, json, statistics; sys.path.insert(0, "/work"); sys.path.insert(0, "/work/e3pkg"); sys.path.insert(0, "/work/kernels")
import torch, bench_layer as B
from pathlib import Path
from e3 import runtime as e3rt
import exl3mm as X

CUBIN = os.environ.get("EXL3MM_CUBIN", "/work/kernels/exl3mm.cubin")
Ms = [int(v) for v in os.environ.get("MS", "64,1024,2048,4096").split(",")]
NG = int(os.environ["NG"]) if "NG" in os.environ else None
ROUTINGS = os.environ.get("ROUTINGS", "router,distinct").split(",")
TIMING = os.environ.get("TIMING", "1") == "1"
OUT = os.environ.get("OUTFILE", "/work/out/stage1.jsonl")

m, layer, ref, info = B.load_layer()
e3rt.bind(layer)
binding = layer.glm6_e3_binding
dev = B.DEV
mod = X.Module(CUBIN)
e3mod = e3rt.DeviceModule(Path(e3rt.__file__).with_name("grouped_fragments.cubin"))
r = B.Router(); g = torch.Generator().manual_seed(77)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
mapping = layer.exl3_mixed_trellis["global_to_combined"]
cap = 4096; k = 8
gate = torch.empty(cap * k, H := 6144, dtype=torch.float16, device=dev)
up = torch.empty_like(gate)
inter = torch.empty(cap * k, 512, dtype=torch.float16, device=dev)
e3out = torch.empty(cap, H, dtype=torch.float32, device=dev)
ybuf = torch.empty(cap * k, H, dtype=torch.float16, device=dev)
myout = torch.empty(cap, H, dtype=torch.bfloat16, device=dev)
stream = torch.cuda.current_stream().cuda_stream


def e3_front(x, routes, m_):
    e3mod.launch("gather", (min(1024, (m_ * k + 7) // 8), H // 128),
                 [x, routes["row_token"], routes["row_expert"], binding["gate_suh"], binding["up_suh"], gate, up, routes["num_rows"], H], stream)
    grid_y = routes["seg_expert"].numel()
    segs = [routes[n] for n in ("seg_expert", "seg_row0", "seg_rows", "num_segs")]
    e3mod.launch("gateup", (512 // 128, grid_y),
                 [gate, up, binding["gate"], binding["up"], binding["gate_svh"], binding["up_svh"], binding["down_suh"], inter, *segs, binding["bits"], H, 512, float("inf")],
                 stream, e3rt.SMEM)
    return grid_y, segs


def e3_down(routes, m_, grid_y, segs):
    e3out[:m_].zero_()
    e3mod.launch("down", (H // 256, grid_y),
                 [inter, binding["down"], binding["down_svh"], e3out, routes["row_token"], routes["row_weight"], *segs, binding["bits"], 512, H],
                 stream, e3rt.SMEM)
    return e3out[:m_].to(torch.bfloat16)


def mine(routes, m_):
    X.run(mod, binding, inter, routes, ybuf, myout[:m_], stream, ng=NG)
    return myout[:m_]


rows = []
for M in Ms:
    for routing in ROUTINGS:
        ss = []
        for _ in range(6):
            x = B.make_x(M, g)
            w, ids = r(x) if routing == "router" else B.distinct_route(M, g)
            ss.append((x, w, ids))
        x, w, ids = ss[0]
        routes = X.route_prep(ids, w, mapping, binding["experts"])
        er = e3rt.route_tables(ids, w, mapping, binding["experts"])
        same = all(torch.equal(routes[n], er[n]) for n in ("row_token", "row_expert", "row_weight", "num_rows", "num_segs", "seg_expert", "seg_row0", "seg_rows"))
        grid_y, segs = e3_front(x, routes, M)
        ref_e3 = e3_down(routes, M, grid_y, segs)
        o = mine(routes, M); torch.cuda.synchronize()
        rf = B.ref_moe(ref, x, w, ids)
        # second run determinism
        o2 = mine(routes, M).clone(); torch.cuda.synchronize()
        d = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
        row = dict(M=M, routing=routing, routes_equal_e3=same, unique_experts=int((routes["seg_rows"] > 0).sum().item()),
                   nsegs=int(routes["num_segs"].item()), ng=NG or X.pick_ng(routes["seg_expert"].numel()),
                   rel_l2_vs_ref=d(o, rf), rel_l2_e3_vs_ref=d(ref_e3, rf), rel_l2_vs_e3=d(o, ref_e3),
                   max_abs_vs_e3=(o.float() - ref_e3.float()).abs().max().item(),
                   max_abs_vs_ref=(o.float() - rf).abs().max().item(), deterministic=bool(torch.equal(o, o2)),
                   nan=bool(torch.isnan(o.float()).any().item()))
        if TIMING:
            def timeit(fn, its=12):
                ts = []
                for it in range(its):
                    xx, ww, ii = ss[it % 6]
                    rt = X.route_prep(ii, ww, mapping, binding["experts"]); gy, sg = e3_front(xx, rt, M)
                    flush.zero_(); torch.cuda.synchronize()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record(); fn(rt, gy, sg); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
                ts.sort(); return statistics.median(ts), ts[0]
            row["us_mine_med"], row["us_mine_min"] = timeit(lambda rt, gy, sg: mine(rt, M))
            row["us_e3down_med"], row["us_e3down_min"] = timeit(lambda rt, gy, sg: e3_down(rt, M, gy, sg))
            # kernel-level split of mine
            def fc2_only(rt, gy, sg):
                nseg = rt["seg_expert"].numel(); ng = NG or X.pick_ng(nseg)
                mod.launch("exl3mm_fc2", (ng, nseg), 512, [inter, binding["down"], binding["down_svh"], binding["bits"], rt["seg_expert"], rt["seg_row0"], rt["seg_rows"], rt["num_segs"], rt["row_weight"], ybuf], stream, mod.fc2_smem)
            row["us_fc2_med"], _ = timeit(fc2_only)
        print(json.dumps(row), flush=True); rows.append(row)
with open(OUT, "w") as f:
    f.write("\n".join(json.dumps(x) for x in rows) + "\n")
