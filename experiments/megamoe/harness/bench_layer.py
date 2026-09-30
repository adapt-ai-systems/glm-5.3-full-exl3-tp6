"""One-layer bench of the LIVE EXL3 MoE path (b12x mixed_trellis one-grid) on one GB10 worker node.

Runs inside the glm53-six:e3-v2-attn-l2-v1 container with the live overlay exl3.py
bind-mounted. Builds a fake RoutedExperts layer holding one TP4 rank slice per expert
from the TP6 exact-fragments checkpoint, then drives the REAL Exl3MoEMethod
_prepare_mixed_rank_sliced_weights / _mixed_rank_sliced_runtime / _apply_mixed_rank_sliced.
"""
import json, os, re, sys, time
from types import SimpleNamespace

import torch

os.environ.setdefault("VLLM_EXL3_PREFILL_CAPACITY", "4096")
os.environ.setdefault("GLM53_MIXED_PREFILL_CHUNK", "1024")
os.environ.setdefault("B12X_COMPILE_CACHE_DIR", "/work/cache")
os.environ.setdefault("VLLM_EXL3_EXT_PATH", "/usr/local/lib/python3.12/dist-packages")

import vllm.model_executor.layers.quantization.exl3 as exl3  # noqa: E402  (the overlay)
from safetensors import safe_open  # noqa: E402

L = int(os.environ.get("BENCH_LAYER", "40"))
E, H, I, TOPK = 256, 6144, 512, 8
MODEL = "/model"
DEV = torch.device("cuda:0")


def _param(shards, tensors, backing):
    return SimpleNamespace(exl3_shard_ids=shards, exl3_tensors=tensors, exl3_backing=backing)


def load_layer(max_batched_tokens=4096):
    """Return (method, layer, ref, info). ref = per-expert tensors for the reference path."""
    tb = json.load(open("/work/tier_bitmap.json"))[str(L)]["k"]
    f = safe_open(f"{MODEL}/model-layer-{L:03d}.safetensors", "pt", device="cpu")
    avail = {}
    for k in f.keys():
        m = re.match(rf"model\.layers\.{L}\.mlp\.experts\.(\d+)\.gate_proj\.rank(\d)\.trellis$", k)
        if m:
            avail[int(m[1])] = int(m[2])
    by_k = {3: sorted(e for e in avail if tb[e] == 3), 4: sorted(e for e in avail if tb[e] == 4)}
    src, subst = {}, {}
    used = {3: 0, 4: 0}
    for e in range(E):
        if e in avail:
            src[e] = e
        else:
            k = tb[e]
            src[e] = by_k[k][used[k] % len(by_k[k])]
            used[k] += 1
            subst[e] = src[e]

    def get(e, proj, fld):
        s = src[e]
        return f.get_tensor(f"model.layers.{L}.mlp.experts.{s}.{proj}_proj.rank{avail[s]}.{fld}")

    proj_of = {"w1": "gate", "w3": "up", "w2": "down"}
    tens = {n: {} for n in ("w13_trellis", "w2_trellis", "w13_mcg", "w2_mcg")}
    gsuh = torch.empty(2, E, H, dtype=torch.float16, device=DEV)      # [shard, E, H]
    gsvh = torch.empty(2, E, I, dtype=torch.float16, device=DEV)
    dsuh = torch.empty(E, I, dtype=torch.float16, device=DEV)
    dsvh = torch.empty(E, H, dtype=torch.float16, device=DEV)
    for e in range(E):
        for si, sh in enumerate(("w1", "w3")):
            p = proj_of[sh]
            tens["w13_trellis"][(e, sh)] = get(e, p, "trellis").to(DEV)
            tens["w13_mcg"][(e, sh)] = get(e, p, "mcg").to(DEV)
            gsuh[si, e] = get(e, p, "suh").to(DEV)
            gsvh[si, e] = get(e, p, "svh").to(DEV)
        tens["w2_trellis"][(e, "w2")] = get(e, "down", "trellis").to(DEV)
        tens["w2_mcg"][(e, "w2")] = get(e, "down", "mcg").to(DEV)
        dsuh[e] = get(e, "down", "suh").to(DEV)
        dsvh[e] = get(e, "down", "svh").to(DEV)

    def views(backing, shards):
        return {(e, sh): (backing[i, e] if len(shards) > 1 else backing[e])
                for e in range(E) for i, sh in enumerate(shards)}

    w13s, w2s = ("w1", "w3"), ("w2",)
    layer = SimpleNamespace(
        layer_name=f"model.layers.{L}.mlp.experts", local_num_experts=E, hidden_size=H,
        exl3_hidden_size=H, exl3_intermediate_size_per_partition=I,
        exl3_layer_bitrates=tuple(int(k) for k in tb), exl3_max_num_batched_tokens=max_batched_tokens,
        activation=exl3.MoEActivation.SILU, exl3_tp_rank=0, exl3_tp_size=4,
        w13_trellis=_param(w13s, dict(tens["w13_trellis"]), None),
        w2_trellis=_param(w2s, dict(tens["w2_trellis"]), None),
        w13_suh=_param(w13s, views(gsuh, w13s), gsuh), w13_svh=_param(w13s, views(gsvh, w13s), gsvh),
        w2_suh=_param(w2s, views(dsuh, w2s), dsuh), w2_svh=_param(w2s, views(dsvh, w2s), dsvh),
        w13_mcg=_param(w13s, tens["w13_mcg"], None), w2_mcg=_param(w2s, tens["w2_mcg"], None),
        w13_mul1=_param(w13s, {}, None), w2_mul1=_param(w2s, {}, None),
    )
    # reference keeps its own dict copies (prepare clears the layer's dicts)
    ref = {"w13_trellis": dict(tens["w13_trellis"]), "w2_trellis": dict(tens["w2_trellis"]),
           "w13_mcg": dict(tens["w13_mcg"]), "w2_mcg": dict(tens["w2_mcg"]),
           "w13_suh": views(gsuh, w13s), "w13_svh": views(gsvh, w13s),
           "w2_suh": views(dsuh, w2s), "w2_svh": views(dsvh, w2s)}
    method = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
    method.quant_config = SimpleNamespace()
    method._prepare_mixed_rank_sliced_weights(layer)
    info = {"layer": L, "substituted": subst, "n_substituted": len(subst), "n_native": len(avail),
            "k": tb, "rank_parts": avail, "src": src}
    return method, layer, ref, info


def ref_moe(ref, x, weights, ids):
    """The overlay's slow per-expert fallback loop (Exl3MoEMethod.apply generic path),
    driven off the per-expert dicts."""
    x2 = x.to(torch.float16)
    w = weights.to(torch.float16)
    out = torch.zeros(x2.shape[0], H, dtype=torch.float32, device=x.device)
    def gemm(group, xx, e, sh):
        key = (e, sh)
        return exl3._exl3_gemm(xx, ref[f"{group}_trellis"][key], ref[f"{group}_suh"][key],
                               ref[f"{group}_svh"][key], key in ref[f"{group}_mcg"], False)
    for e in range(E):
        pos = (ids == e).nonzero(as_tuple=False)
        if pos.shape[0] == 0:
            continue
        t, r = pos[:, 0], pos[:, 1]
        xi = x2.index_select(0, t)
        gate = gemm("w13", xi, e, "w1")
        up = gemm("w13", xi, e, "w3")
        hid = torch.nn.functional.silu(gate) * up
        y = gemm("w2", hid, e, "w2")
        out.index_add_(0, t, (y * w[t, r].unsqueeze(-1)).to(torch.float32))
    return out


class Router:
    def __init__(self):
        f = safe_open(f"{MODEL}/model-layer-{L:03d}.safetensors", "pt", device="cpu")
        self.w = f.get_tensor(f"model.layers.{L}.mlp.gate.weight").to(DEV)
        self.b = f.get_tensor(f"model.layers.{L}.mlp.gate.e_score_correction_bias").to(DEV)

    def __call__(self, x):
        s = torch.sigmoid((x.float() @ self.w.float().T))
        ids = torch.topk(s + self.b, TOPK, dim=-1).indices
        w = s.gather(1, ids)
        w = w / w.sum(-1, keepdim=True) * 2.5
        return w.float().contiguous(), ids.contiguous()  # ids int64 like live topk_indices_dtype


def distinct_route(M, gen):
    """Worst case: every (token, k) hits a distinct expert when M*8<=256, else spread evenly."""
    n = M * TOPK
    perm = torch.randperm(E, generator=gen)
    if n <= E:
        ids = perm[:n].reshape(M, TOPK)
    else:
        ids = torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(M)])
    w = torch.rand(M, TOPK, generator=gen) + 0.1
    w = w / w.sum(-1, keepdim=True) * 2.5
    return w.float().to(DEV).contiguous(), ids.to(DEV).contiguous()


def make_x(M, gen):
    return (torch.randn(M, H, generator=gen) * 0.5).to(torch.bfloat16).to(DEV).contiguous()


def expert_bytes(k):
    """Weight bytes for one expert TP4 slice: 3 mats of 512x6144 at k bits + rotations."""
    trellis = 3 * I * H * k // 8
    rot = (2 * H + 2 * I + I + H) * 2
    return trellis, rot
