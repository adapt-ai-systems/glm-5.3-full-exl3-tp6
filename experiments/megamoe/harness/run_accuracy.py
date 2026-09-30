import sys, json; sys.path.insert(0, "/work")
import torch, bench_layer as B
m, layer, ref, info = B.load_layer()
json.dump({k: v for k, v in info.items() if k != "k"}, open("/work/out/layer_info.json", "w"))
r = B.Router()
g = torch.Generator().manual_seed(7)
rows = []
for M in (1, 5, 20, 1024, 4096):
    for kind in ("router", "distinct"):
        x = B.make_x(M, g)
        w, ids = r(x) if kind == "router" else B.distinct_route(M, g)
        out = m._apply_mixed_rank_sliced(layer, x, w, ids).float()
        ro = B.ref_moe(ref, x, w, ids)
        torch.cuda.synchronize()
        d = out - ro
        floor = (ro.to(torch.bfloat16).float() - ro)  # bf16 output rounding noise floor
        row = dict(M=M, routing=kind, unique_experts=int(ids.unique().numel()),
                   max_abs_err=d.abs().max().item(), ref_absmax=ro.abs().max().item(),
                   rel_l2=(d.norm() / ro.norm()).item(), bf16_floor_rel_l2=(floor.norm() / ro.norm()).item())
        print(row, flush=True); rows.append(row)
with open("/work/out/accuracy.jsonl", "w") as f:
    for row in rows: f.write(json.dumps(row) + "\n")
