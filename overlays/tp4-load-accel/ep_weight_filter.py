# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filter out non-local expert weights during loading to avoid redundant I/O.

In DP+EP deployments each rank only needs its own expert shard.  Skipping
non-local expert tensors *before* they are read from disk eliminates the
majority of storage I/O for MoE models (experts typically account for
~85-90 % of total weight bytes).
"""

import regex as re

# Matches per-expert weight names like ".experts.42.gate_proj.weight".
# Does NOT match 3D fused-expert names like ".experts.gate_proj.weight"
# (no numeric id) — those are intentionally left unfiltered so the full
# tensor is loaded and sliced later by RoutedExperts.weight_loader.
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")


def parse_expert_id(weight_name: str) -> int | None:
    """Return the expert id embedded in *weight_name*, or ``None`` if it is
    not an per-expert weight.

    Returns ``None`` for dense weights (attention, layernorm, embedding),
    shared experts, and 3D fused-expert tensors where all experts are stored
    in a single tensor without a numeric expert id in the name."""
    m = _EXPERT_ID_RE.search(weight_name)
    return int(m.group(1)) if m else None


def compute_local_expert_ids(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    placement: str = "linear",
) -> set[int] | None:
    """Compute the set of global expert ids owned by *ep_rank*.

    Returns ``None`` when EP is not active (``ep_size <= 1``), meaning all
    experts are local and no filtering should be performed.

    The distribution logic mirrors
    :func:`vllm.model_executor.layers.fused_moe.layer.determine_expert_map`.

    Args:
        placement: ``"linear"`` for contiguous assignment,
            ``"round_robin"`` for interleaved assignment.
    """
    if ep_size <= 1:
        return None

    if placement == "linear":
        base = num_experts // ep_size
        remainder = num_experts % ep_size
        start = ep_rank * base + min(ep_rank, remainder)
        local_count = base + (1 if ep_rank < remainder else 0)
        return set(range(start, start + local_count))
    elif placement == "round_robin":
        return set(range(ep_rank, num_experts, ep_size))
    else:
        raise ValueError(f"Unknown expert placement strategy: {placement}")


def should_skip_weight(
    weight_name: str,
    local_expert_ids: set[int] | None,
) -> bool:
    """Return ``True`` if *weight_name* is an expert weight that does not
    belong to the local rank and should be skipped during loading."""
    if local_expert_ids is None:
        return False
    eid = parse_expert_id(weight_name)
    if eid is None:
        # Not an expert weight (dense / shared-expert / embedding) → keep.
        return False
    # Only skip heavy weight tensors, never scale/metadata tensors.
    # Scale tensors are tiny and some backends need them from ALL experts
    # (e.g. FlashInfer NVFP4 computes a global max of activation scales).
    if not weight_name.endswith((".weight", ".weight_packed")):
        return False
    return eid not in local_expert_ids
# TR3 REC1 wrapper
# Keep the upstream prefix unchanged; all non-eligible cases use its predicate.
import json
import os
from functools import lru_cache

_tr3_original_should_skip_weight = should_skip_weight
_TR3_CONFIG_PATH = "/model/config.json"
_TR3_WEIGHT_RE = re.compile(
    r"model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>[0-3])\.(?P<field>trellis|suh|svh|mcg)"
)
_TR3_TENSOR_SCHEMA = (
    "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
)


@lru_cache(maxsize=1)
def _tr3_checkpoint_eligible(pid: int) -> bool:
    """Read the fixed launcher checkpoint once per process, including after fork."""
    del pid  # Part of the cache key, not checkpoint metadata.
    try:
        with open(_TR3_CONFIG_PATH, encoding="utf-8") as source:
            config = json.load(source)
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    metadata = config.get("hybrid_tr3_tail")
    if not isinstance(metadata, dict):
        return False
    layers = metadata.get("moe_layers")
    return (
        metadata.get("format") == "exl3-trellis"
        and type(metadata.get("tp")) is int
        and metadata["tp"] == 4
        and metadata.get("codebook") == "mcg"
        and isinstance(layers, list)
        and all(type(layer) is int for layer in layers)
        and layers == [3, 78]
        and type(metadata.get("experts_per_layer")) is int
        and metadata["experts_per_layer"] == 256
        and metadata.get("tensor_schema") == _TR3_TENSOR_SCHEMA
        and metadata.get("rotation_layout", "per_expert_v1") == "per_expert_v1"
        and metadata.get("shared_h_tensor_schema") is None
    )


def _tr3_current_rank() -> int | None:
    """Use the launcher's TP rank, or lazily resolve an initialized TP group."""
    rank_env = os.environ.get("TR3_RANK")
    if rank_env is not None:
        return int(rank_env) if rank_env in ("0", "1", "2", "3") else None
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:
        # An unavailable/uninitialized group must not discard any extra names.
        return None
    return rank if type(rank) is int and 0 <= rank <= 3 else None


def should_skip_weight(
    weight_name: str,
    local_expert_ids: set[int] | None,
) -> bool:
    """Skip only foreign TP4 per-expert components before lazy get_tensor."""
    if os.environ.get("TR3_SKIP_FOREIGN_RANKS") == "1":
        match = _TR3_WEIGHT_RE.fullmatch(weight_name)
        if match is not None:
            try:
                in_bounds = (
                    3 <= int(match["layer"]) <= 78
                    and 0 <= int(match["expert"]) < 256
                )
            except ValueError:
                in_bounds = False
            if in_bounds and _tr3_checkpoint_eligible(os.getpid()):
                rank = _tr3_current_rank()
                if rank is not None and int(match["rank"]) != rank:
                    return True
    return _tr3_original_should_skip_weight(weight_name, local_expert_ids)


# TP4 fastload (TR3_FAST=1) wrapper: also skip, BEFORE get_tensor, names whose tensor the consuming model would
# discard anyway. Off (or any doubt) -> the REC1 predicate above, unchanged.
#   a) own-rank per-expert components of a layer restored from the prepared cache (load_exl3_weight returns early on
#      _tr3_cache_skip for exactly these: w13/w2 suh/svh/trellis/mcg of a HIT layer);
#   b) target pass: model.layers.<spec>.* (DeepseekV2 load_weights: `if spec_layer is not None: continue`);
#   c) drafter pass: everything not under model.layers.<spec>. (DeepSeekMTP load_weights: `if spec_layer is None:
#      continue`). The drafter pass is recognised only once the spec layer's MoE has been constructed AND every
#      target HIT layer has been restored (target process_weights_after_loading is done) - the observed order.
_tr3_rec1_should_skip_weight = should_skip_weight
_TR3_FAST = os.environ.get("TR3_FAST") == "1"
_TR3_SPEC_LAYER = 78
_TR3_LAYER_RE = re.compile(r"model\.layers\.(?P<layer>[0-9]+)\.")
_tr3_fast_counts = {"cache_owned": 0, "spec_in_target": 0, "target_in_drafter": 0}


def _tr3_fast_phase():
    """None (unknown -> no extra skipping), 'target' or 'drafter'."""
    import sys
    rt = sys.modules.get("tr3_runtime")
    if rt is None or not getattr(rt, "FAST", False):
        return None, None
    constructed, hit, restored = rt.CONSTRUCTED, rt.HIT, rt.RESTORED
    if not constructed:
        return None, rt
    if _TR3_SPEC_LAYER not in constructed:
        return "target", rt
    target_hits = hit - {_TR3_SPEC_LAYER}
    if target_hits and target_hits <= restored:
        return "drafter", rt
    return None, rt


def should_skip_weight(
    weight_name: str,
    local_expert_ids: set[int] | None,
) -> bool:
    if _TR3_FAST and _tr3_checkpoint_eligible(os.getpid()):
        phase, rt = _tr3_fast_phase()
        if phase is not None:
            m = _TR3_LAYER_RE.match(weight_name)
            layer = int(m["layer"]) if m else None
            if phase == "target" and layer == _TR3_SPEC_LAYER:
                _tr3_fast_counts["spec_in_target"] += 1
                return True
            if phase == "drafter" and layer != _TR3_SPEC_LAYER:
                _tr3_fast_counts["target_in_drafter"] += 1
                return True
            e = _TR3_WEIGHT_RE.fullmatch(weight_name)
            if e is not None and int(e["layer"]) in rt.HIT and _tr3_current_rank() == int(e["rank"]):
                _tr3_fast_counts["cache_owned"] += 1
                return True
    return _tr3_rec1_should_skip_weight(weight_name, local_expert_ids)
