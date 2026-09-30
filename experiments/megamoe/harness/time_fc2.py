"""Time exl3mm fc2 (and finalize) alone for variant cubins. usage: time_fc2.py cubin[,cubin...] (env MS, NG)"""
import sys, os, json, statistics; sys.path.insert(0, "/work"); sys.path.insert(0, "/work/e3pkg"); sys.path.insert(0, "/work/kernels")
import torch, bench_layer as B
from pathlib import Path
from e3 import runtime as e3rt
import exl3mm as X
cubins = sys.argv[1].split(","); Ms = [int(v) for v in os.environ.get("MS", "1024,4096").split(",")]
NG = int(os.environ["NG"]) if "NG" in os.environ else None
m, layer, ref, info = B.load_layer(); e3rt.bind(layer); binding = layer.glm6_e3_binding
dev = B.DEV; mods = {c: X.Module(c, int(os.environ.get("SMEM", X.FC2_SMEM))) for c in cubins}
e3mod = e3rt.DeviceModule(Path(e3rt.__file__).with_name("grouped_fragments.cubin"))
r = B.Router(); g = torch.Generator().manual_seed(5)
mapping = layer.exl3_mixed_trellis["global_to_combined"]; H = 6144; k = 8
gate = torch.empty(4096 * k, H, dtype=torch.float16, device=dev); up = torch.empty_like(gate)
inter = torch.empty(4096 * k, 512, dtype=torch.float16, device=dev)
ybuf = torch.empty(4096 * k, H, dtype=torch.float16, device=dev); out = torch.empty(4096, H, dtype=torch.bfloat16, device=dev)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
stream = torch.cuda.current_stream().cuda_stream
for M in Ms:
    x = B.make_x(M, g); w, ids = r(x)
    routes = X.route_prep(ids, w, mapping, binding["experts"])
    e3mod.launch("gather", (min(1024, (M * k + 7) // 8), H // 128), [x, routes["row_token"], routes["row_expert"], binding["gate_suh"], binding["up_suh"], gate, up, routes["num_rows"], H], stream)
    segs = [routes[n] for n in ("seg_expert", "seg_row0", "seg_rows", "num_segs")]
    e3mod.launch("gateup", (4, routes["seg_expert"].numel()), [gate, up, binding["gate"], binding["up"], binding["gate_svh"], binding["up_svh"], binding["down_suh"], inter, *segs, binding["bits"], H, 512, float("inf")], stream, e3rt.SMEM)
    for c, mod in mods.items():
        res = {}
        for which in ("fc2", "both"):
            ts = []
            for it in range(15):
                if os.environ.get("HOT") == "1":
                    e3mod.launch("gather", (min(1024, (M * k + 7) // 8), H // 128), [x, routes["row_token"], routes["row_expert"], binding["gate_suh"], binding["up_suh"], gate, up, routes["num_rows"], H], stream)
                    e3mod.launch("gateup", (4, routes["seg_expert"].numel()), [gate, up, binding["gate"], binding["up"], binding["gate_svh"], binding["up_svh"], binding["down_suh"], inter, *segs, binding["bits"], H, 512, float("inf")], stream, e3rt.SMEM)
                if os.environ.get("POISON") == "1": ybuf.fill_(float("nan"))
                flush.zero_(); torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                if which == "fc2":
                    nseg = routes["seg_expert"].numel(); ng = NG or X.pick_ng(nseg)
                    mod.launch("exl3mm_fc2", (ng, nseg), int(os.environ.get("THREADS", 512)), [inter, binding["down"], binding["down_svh"], binding["bits"], routes["seg_expert"], routes["seg_row0"], routes["seg_rows"], routes["num_segs"], routes["row_weight"], ybuf], stream, mod.fc2_smem)
                else:
                    X.run(mod, binding, inter, routes, ybuf, out[:M], stream, ng=NG, threads=int(os.environ.get("THREADS", 512)))
                e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
                if os.environ.get("POISON") == "1" and which == "fc2":
                    nr = int(routes["num_rows"].item())
                    fin = torch.isfinite(ybuf[:nr]).all().item()
                    assert fin, f"NaN left in ybuf rows [0,{nr}) after timed fc2 call"
                    poison_ok = True
            ts.sort(); res[which] = (round(statistics.median(ts)), round(ts[0]))
        print(json.dumps(dict(poison_all_finite=os.environ.get("POISON")=="1", cubin=Path(c).name, M=M, fc2_med_min=res["fc2"], both_med_min=res["both"])), flush=True)
