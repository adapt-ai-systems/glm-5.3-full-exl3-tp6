import json, torch
d = torch.device("cuda:0"); res = {}
def best(fn, nbytes, n=20, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) / 1e3)
    return dict(best_GBps=round(nbytes / min(ts) / 1e9, 1), median_GBps=round(nbytes / sorted(ts)[len(ts)//2] / 1e9, 1))
for gb in (1, 2):
    n = gb << 30
    b = torch.empty(n, dtype=torch.uint8, device=d)
    res[f"{gb}GB_zero_"] = best(lambda: b.zero_(), n)
    res[f"{gb}GB_fill_fp16"] = best(lambda: b.view(torch.float16).fill_(1.0), n)
    src = torch.randint(0, 255, (n,), dtype=torch.uint8, device=d)
    r = best(lambda: b.copy_(src), 2 * n); res[f"{gb}GB_copy_rw"] = r
    del b, src
print(json.dumps(res, indent=1)); json.dump(res, open("/work/out/wrbw.json", "w"), indent=1)
