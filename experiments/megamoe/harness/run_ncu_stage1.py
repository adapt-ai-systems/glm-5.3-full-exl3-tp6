"""ncu target: e3 front (gather+gateup) then exl3mm fc2+finalize. usage: run_ncu_stage1.py M [cubin]"""
import sys, os; sys.path.insert(0, "/work"); sys.path.insert(0, "/work/e3pkg"); sys.path.insert(0, "/work/kernels")
import torch, bench_layer as B
from pathlib import Path
from e3 import runtime as e3rt
import exl3mm as X
M = int(sys.argv[1]); cubin = sys.argv[2] if len(sys.argv) > 2 else "/work/kernels/exl3mm.cubin"
m, layer, ref, info = B.load_layer(); e3rt.bind(layer); binding = layer.glm6_e3_binding
dev = B.DEV; mod = X.Module(cubin); e3mod = e3rt.DeviceModule(Path(e3rt.__file__).with_name("grouped_fragments.cubin"))
r = B.Router(); g = torch.Generator().manual_seed(3); x = B.make_x(M, g); w, ids = r(x)
mapping = layer.exl3_mixed_trellis["global_to_combined"]; H = 6144; k = 8
gate = torch.empty(4096 * k, H, dtype=torch.float16, device=dev); up = torch.empty_like(gate)
inter = torch.empty(4096 * k, 512, dtype=torch.float16, device=dev)
ybuf = torch.empty(4096 * k, H, dtype=torch.float16, device=dev); out = torch.empty(4096, H, dtype=torch.bfloat16, device=dev)
stream = torch.cuda.current_stream().cuda_stream
routes = X.route_prep(ids, w, mapping, binding["experts"])
e3mod.launch("gather", (min(1024, (M * k + 7) // 8), H // 128), [x, routes["row_token"], routes["row_expert"], binding["gate_suh"], binding["up_suh"], gate, up, routes["num_rows"], H], stream)
segs = [routes[n] for n in ("seg_expert", "seg_row0", "seg_rows", "num_segs")]
e3mod.launch("gateup", (4, routes["seg_expert"].numel()), [gate, up, binding["gate"], binding["up"], binding["gate_svh"], binding["up_svh"], binding["down_suh"], inter, *segs, binding["bits"], H, 512, float("inf")], stream, e3rt.SMEM)
for _ in range(3): X.run(mod, binding, inter, routes, ybuf, out[:M], stream)
torch.cuda.synchronize(); print("done")
