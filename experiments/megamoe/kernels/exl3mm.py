"""Host side for exl3mm stage 1: cubin loader + own route prep (torch) + launch helpers."""
import ctypes as C
from pathlib import Path
import torch

H, I = 6144, 512
FC2_SMEM = 99328
TILE_COLS = 256


class Module:
    def __init__(self, cubin, fc2_smem=FC2_SMEM):
        self.driver = C.CDLL("libcuda.so.1")
        self._chk(self.driver.cuInit(0))
        self.module = C.c_void_p()
        self._chk(self.driver.cuModuleLoad(C.byref(self.module), str(cubin).encode()))
        self.fn = {}
        for name in ("exl3mm_fc2", "exl3mm_finalize"):
            f = C.c_void_p()
            self._chk(self.driver.cuModuleGetFunction(C.byref(f), self.module, name.encode()))
            self.fn[name] = f
        self._chk(self.driver.cuFuncSetAttribute(self.fn["exl3mm_fc2"], 8, fc2_smem))
        self.fc2_smem = fc2_smem

    @staticmethod
    def _chk(c):
        if c:
            raise RuntimeError(f"CUDA driver error {c}")

    def launch(self, name, grid, block, args, stream, smem=0):
        st = [C.c_void_p(a.data_ptr()) if hasattr(a, "data_ptr") else C.c_int(a) for a in args]
        params = (C.c_void_p * len(st))(*[C.addressof(v) for v in st])
        self._chk(self.driver.cuLaunchKernel(self.fn[name], C.c_uint(grid[0]), C.c_uint(grid[1]), C.c_uint(1),
                                             C.c_uint(block), C.c_uint(1), C.c_uint(1), C.c_uint(smem),
                                             C.c_void_p(stream), params, None))


def route_prep(ids, weights, mapping, experts, seg_rows=64):
    """Expert-sorted route tables (same semantics as E3's route_tables) plus pos[t*8+k] -> sorted row."""
    m, k = ids.shape
    dev = ids.device
    local = mapping[ids.to(torch.long)].reshape(-1).to(torch.long)
    keys = torch.where(local >= 0, local, torch.full_like(local, experts))
    skeys, order = torch.sort(keys, stable=True)
    counts = torch.zeros(experts + 1, dtype=torch.int32, device=dev)
    counts.scatter_add_(0, keys, torch.ones_like(keys, dtype=torch.int32))
    counts = counts[:experts]
    cum = counts.cumsum(0)
    off = cum - counts
    tiles = (counts + seg_rows - 1) // seg_rows
    tcum = tiles.cumsum(0)
    toff = tcum - tiles
    seg = torch.arange((m * k + seg_rows - 1) // seg_rows + experts, device=dev)
    exp = torch.searchsorted(tcum, seg, right=True).clamp(max=experts - 1)
    tile = seg - toff[exp]
    ar = torch.arange(m * k, device=dev, dtype=torch.int32)
    pos = torch.empty(m * k, dtype=torch.int32, device=dev)
    pos.scatter_(0, order, torch.where(skeys < experts, ar, torch.full_like(ar, -1)))
    return dict(row_token=(order // k).contiguous(), row_expert=skeys.clamp(max=experts - 1).to(torch.int32),
                row_weight=weights.reshape(-1)[order].to(torch.float32),
                num_rows=cum[-1:].to(torch.int32), num_segs=tcum[-1:].to(torch.int32),
                seg_expert=exp.to(torch.int32), seg_row0=(off[exp] + tile * seg_rows).to(torch.int32),
                seg_rows=(counts[exp] - tile * seg_rows).clamp(min=0, max=seg_rows).to(torch.int32),
                pos=pos)


def pick_ng(nsegs):
    """column groups per segment: divide 24 tiles, aim for >= ~192 CTAs."""
    for ng in (1, 2, 4, 6, 8, 12, 24):
        if nsegs * ng >= 384:
            return ng
    return 24


def run(mod, binding, inter, routes, ybuf, out, stream, ng=None, threads=512):
    nseg = routes["seg_expert"].numel()
    ng = ng or pick_ng(nseg)
    mod.launch("exl3mm_fc2", (ng, nseg), threads,
               [inter, binding["down"], binding["down_svh"], binding["bits"], routes["seg_expert"], routes["seg_row0"],
                routes["seg_rows"], routes["num_segs"], routes["row_weight"], ybuf], stream, mod.fc2_smem)
    m = out.shape[0]
    mod.launch("exl3mm_finalize", (m, 1), 256, [ybuf, routes["pos"], routes["num_rows"], out], stream)
