import torch, json, os
d = torch.device("cuda:0")
def best(fn, flops, n=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) / 1e3)
    return flops / min(ts) / 1e12, flops / sorted(ts)[len(ts)//2] / 1e12
rows = []
def add(name, M, N, K, dtype, fn):
    b, m = best(fn, 2 * M * N * K)
    rows.append(dict(op=name, M=M, N=N, K=K, dtype=dtype, best_TFLOPs=b, median_TFLOPs=m)); print(rows[-1], flush=True)
for dt in (torch.float16, torch.bfloat16):
    n = str(dt).split(".")[-1]
    a = torch.randn(8192, 8192, device=d, dtype=dt); b = torch.randn(8192, 8192, device=d, dtype=dt)
    add("mm", 8192, 8192, 8192, n, lambda: a @ b)
    for M in (128, 256, 512, 1024, 4096):
        for (K, N) in ((6144, 1024), (512, 6144)):
            a = torch.randn(M, K, device=d, dtype=dt); w = torch.randn(K, N, device=d, dtype=dt)
            add("mm", M, N, K, n, lambda a=a, w=w: a @ w)
a = torch.randn(8192, 8192, device=d).to(torch.float8_e4m3fn); b = torch.randn(8192, 8192, device=d).to(torch.float8_e4m3fn).t().contiguous().t()
sc = torch.tensor(1.0, device=d)
try:
    add("scaled_mm", 8192, 8192, 8192, "e4m3", lambda: torch._scaled_mm(a, b, scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16))
except Exception as ex: print("scaled_mm failed", ex)
json.dump(rows, open("/work/out/mm_peak.json", "w"), indent=1)
