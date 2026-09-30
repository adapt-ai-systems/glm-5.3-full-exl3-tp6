import json, torch
d = torch.device("cuda:0")
res = {}
def best(fn, nbytes, n=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) / 1e3)
    return dict(best_GBps=nbytes / min(ts) / 1e9, median_GBps=nbytes / sorted(ts)[len(ts)//2] / 1e9)
flush = torch.empty(256 << 20, dtype=torch.uint8, device=d)
for gb in (2, 4):
    n = gb << 30
    buf = torch.randint(0, 255, (n,), dtype=torch.uint8, device=d)
    for name, view in (("sum_int64", buf.view(torch.int64)), ("sum_fp16", buf.view(torch.float16)), ("sum_bf16", buf.view(torch.bfloat16))):
        res[f"{gb}GB_{name}"] = best(lambda v=view: v.sum(dtype=torch.float32 if v.dtype != torch.int64 else torch.int64), n)
    # 2 GB device->device copy counts read+write
    dst = torch.empty_like(buf)
    r = best(lambda: dst.copy_(buf), n)
    res[f"{gb}GB_copy_read_only_equiv"] = r
    res[f"{gb}GB_copy_rw_GBps"] = {k: v * 2 for k, v in r.items()}
    del dst, buf
# custom vectorized read kernel via torch.compile-free trick: amax over uint8 viewed int32 rows
buf = torch.randint(0, 255, (4 << 30,), dtype=torch.uint8, device=d)
v = buf.view(torch.int32).view(-1, 4096)
res["4GB_rowmax_int32"] = best(lambda: v.max(dim=1).values, 4 << 30)
print(json.dumps(res, indent=1))
json.dump(res, open("/work/out/membw.json", "w"), indent=1)
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw", "--format=csv"], capture_output=True, text=True).stdout)
