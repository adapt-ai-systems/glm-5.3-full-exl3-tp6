"""Target for ncu: python run_ncu_target.py e3 4096 | b12x 5"""
import sys, os; sys.path.insert(0, "/work"); sys.path.insert(0, "/work/e3pkg")
import torch, bench_layer as B
which, M = sys.argv[1], int(sys.argv[2])
m, layer, ref, info = B.load_layer()
r = B.Router(); g = torch.Generator().manual_seed(3)
x = B.make_x(M, g); w, ids = r(x)
if which == "e3":
    from e3 import runtime as e3rt
    e3rt.bind(layer); f = lambda: e3rt.apply(layer, x, w, ids, stream_scratch=True)
else:
    f = lambda: m._apply_mixed_rank_sliced(layer, x, w, ids)
for _ in range(2): f()
torch.cuda.synchronize()
f(); torch.cuda.synchronize()
print("done")
