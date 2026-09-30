"""Decode graph-replay timing vs VLLM_EXL3_TRELLIS_MAX_M. usage: run_decode_knob.py M1,M2,.. (env VLLM_EXL3_TRELLIS_MAX_M, NSET)
routings: router, same8 (all tokens routed onto the same 8 experts -> 8 unique experts total)."""
import sys, json, statistics, os; sys.path.insert(0, "/work")
import torch, bench_layer as B
NSET = int(os.environ.get("NSET", "24")); MAXM = os.environ.get("VLLM_EXL3_TRELLIS_MAX_M", "32")
m, layer, ref, info = B.load_layer(); r = B.Router(); g = torch.Generator().manual_seed(11)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=B.DEV)
def same8(M):
    base = torch.randperm(B.E, generator=g)[:B.TOPK]
    ids = torch.stack([base[torch.randperm(B.TOPK, generator=g)] for _ in range(M)])
    w = torch.rand(M, B.TOPK, generator=g) + 0.1; w = w / w.sum(-1, keepdim=True) * 2.5
    return w.float().to(B.DEV).contiguous(), ids.to(B.DEV).contiguous()
for M in [int(v) for v in sys.argv[1].split(",")]:
    for kind in os.environ.get("KINDS", "router,same8").split(","):
        ss = []
        for _ in range(NSET):
            x = B.make_x(M, g); w, ids = r(x) if kind == "router" else same8(M); ss.append((x, w, ids))
        xs, ws, ids_s = (t.clone() for t in ss[0])
        m._apply_mixed_rank_sliced(layer, xs, ws, ids_s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr): out = m._apply_mixed_rank_sliced(layer, xs, ws, ids_s)
        ts = []
        for it in range(NSET * 3):
            x, w, ids = ss[it % NSET]; xs.copy_(x); ws.copy_(w); ids_s.copy_(ids); flush.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); gr.replay(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
        ts.sort(); ue = sum(len(t[2].unique()) for t in ss) / NSET
        print(json.dumps(dict(max_m=int(MAXM), M=M, routing=kind, us_median=round(statistics.median(ts), 1), us_min=round(ts[0], 1), us_p90=round(ts[int(.9 * (len(ts) - 1))], 1), unique_experts=round(ue, 1))), flush=True)
        del gr
