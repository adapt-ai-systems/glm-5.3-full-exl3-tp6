import sys, json, statistics, os; sys.path.insert(0, "/work")
import torch, bench_layer as B
BW = float(os.environ.get("ROOFLINE_GBPS", "243.2"))   # measured best read BW (membw.py)
NSET = int(os.environ.get("NSET", "24"))
m, layer, ref, info = B.load_layer()
K = info["k"]
r = B.Router()
g = torch.Generator().manual_seed(11)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=B.DEV)   # L2 flush: 256 MB write before each timed call


def sets(M, kind):
    out = []
    for _ in range(NSET):
        x = B.make_x(M, g)
        w, ids = r(x) if kind == "router" else B.distinct_route(M, g)
        out.append((x, w, ids))
    return out


def bytes_for(ids):
    u = ids.unique().tolist()
    t = rot = 0
    for e in u:
        a, b = B.expert_bytes(K[e]); t += a; rot += b
    return len(u), t, rot


def summarize(M, kind, mode, ts, ss):
    ts_us = sorted(t * 1e3 for t in ts)   # ms -> us
    med = statistics.median(ts_us)
    stats = [bytes_for(s[2]) for s in ss]
    ue = statistics.mean(s[0] for s in stats); tb = statistics.mean(s[1] for s in stats); rb = statistics.mean(s[2] for s in stats)
    bts = tb + rb
    row = dict(M=M, routing=kind, mode=mode, us_median=med, us_min=ts_us[0], us_mean=statistics.mean(ts_us),
               us_p90=ts_us[int(0.9 * (len(ts_us) - 1))], unique_experts=ue, trellis_bytes=tb, rot_bytes=rb, weight_bytes=bts,
               GBps_median=bts / (med * 1e-6) / 1e9, pct_roofline=100 * bts / (med * 1e-6) / 1e9 / BW,
               floor_us=bts / (BW * 1e9) * 1e6)
    print(json.dumps(row), flush=True)
    return row


rows = []
# ---- decode: CUDA graph replay, M=1..20
for M in list(range(1, 21)):
    for kind in ("router", "distinct"):
        ss = sets(M, kind)
        xs, ws, ids_s = (t.clone() for t in ss[0])
        m._apply_mixed_rank_sliced(layer, xs, ws, ids_s)    # eager warm
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = m._apply_mixed_rank_sliced(layer, xs, ws, ids_s)
        ts = []
        for it in range(NSET * 2):
            x, w, ids = ss[it % NSET]
            xs.copy_(x); ws.copy_(w); ids_s.copy_(ids)
            flush.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); gr.replay(); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e))
        rows.append(summarize(M, kind, "graph", ts, ss))
        del gr
# ---- eager decode extras and prefill (eager)
for M, kinds in ((5, ("router",)), (20, ("router",)), (1024, ("router", "distinct")), (4096, ("router", "distinct"))):
    for kind in kinds:
        ss = sets(M, kind) if M <= 32 else sets(M, kind)[:8]
        for i in range(2): m._apply_mixed_rank_sliced(layer, *ss[0])
        torch.cuda.synchronize()
        ts = []
        for it in range(len(ss) * (2 if M <= 32 else 1)):
            x, w, ids = ss[it % len(ss)]
            flush.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); m._apply_mixed_rank_sliced(layer, x, w, ids); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e))
        rows.append(summarize(M, kind, "eager", ts, ss))
with open("/work/out/timing.jsonl", "w") as f:
    for row in rows: f.write(json.dumps(row) + "\n")
