# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``.

keys-DCP port (2026-09-01): adds Decode Context Parallel (KV sharded across the TP ranks)
support, mirroring the SM100 ``FlashInferMLASparseImpl`` branch:
  * ``can_return_lse_for_decode = True`` so the CP compatibility check passes and the layer
    combines per-rank partial attention with ``MLADCPManager.combine``;
  * under DCP the global top-k token indices are filtered to this rank's local KV slots with
    ``triton_filter_and_convert_dcp_index`` (interleave-aware);
  * the flashinfer SM120 sparse kernel is asked for the softmax LSE (``return_lse``), which is
    normalized to ``(num_tokens, num_heads)``; rows with no local top-k tokens get output 0 and
    LSE ``-inf`` so they drop out of the cross-rank softmax merge;
  * the output is allocated from ``q.shape[1]`` (all heads after the DCP query all-gather),
    not the layer's TP-local head count.
``EXL3_SM120_LSE_BASE_E`` (default 1) selects the LSE base reported to the combine kernel; the
SM120 sparse kernel follows the FlashMLA (natural-log) convention. Set 0 for base-2.
"""

import os
from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


# The layer's DCP combine reads ``attn_metadata.decode``; the sparse metadata has no such field.
if not hasattr(FlashInferMLASparseMetadata, "decode"):
    FlashInferMLASparseMetadata.decode = None  # type: ignore[attr-defined]


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation (+ DCP)."""

    is_sparse = True
    supports_dense_mha_prefill = False
    # DCP: we can hand the softmax LSE to the cross-rank combine.
    can_return_lse_for_decode = True
    lse_base_on_e = os.environ.get("EXL3_SM120_LSE_BASE_E", "1") == "1"

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.rope_pad = 0
        if self.qk_rope_head_dim == 0:
            if self.kv_lora_rank != 512:
                raise NotImplementedError(
                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "
                    "576-wide GLM_NSA geometry, which requires "
                    f"kv_lora_rank=512; got {self.kv_lora_rank}."
                )
            self.rope_pad = 64
        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None
        self._dcp_logged = False

    @staticmethod
    def _normalize_lse(lse: torch.Tensor, num_tokens: int, num_heads: int) -> torch.Tensor:
        # (num_tokens, num_heads) is what the shared DCP reducer expects; the kernel may
        # hand back (T, H, 1) / (T, 1, H) / (B, q_len, H).
        if lse.dim() == 3:
            if lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            elif lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.shape[0] * lse.shape[1] == num_tokens:
                lse = lse.reshape(num_tokens, lse.shape[-1])
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected SM120 sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.rope_pad:
            q = torch.nn.functional.pad(q, (0, self.rope_pad))

        num_actual_toks = q.shape[0]
        # Under DCP the layer all-gathers queries across the CP ranks first, so q carries
        # every head (not just this rank's TP slice).
        num_heads = q.shape[1]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_dcp = self.dcp_world_size > 1
        if use_dcp:
            if not self._dcp_logged:
                from vllm.logger import init_logger

                init_logger(__name__).info(
                    "[keys-DCP] SM120 sparse MLA: dcp_world=%d rank=%d interleave=%d "
                    "heads=%d lse_base_e=%s",
                    self.dcp_world_size,
                    self.dcp_rank,
                    attn_metadata.cp_kv_cache_interleave_size,
                    num_heads,
                    self.lse_base_on_e,
                )
                self._dcp_logged = True
            topk_indices_physical, topk_lengths = cast(
                tuple[torch.Tensor, torch.Tensor],
                triton_filter_and_convert_dcp_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                ),
            )
        else:
            topk_indices_physical, topk_lengths = cast(
                tuple[torch.Tensor, torch.Tensor],
                triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    return_valid_counts=True,
                ),
            )
        sparse_topk_capacity = topk_indices_physical.shape[1]
        empty_rows = topk_lengths == 0
        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(
            empty_rows, 0
        )
        topk_lengths = topk_lengths.clamp(min=1)

        output = q.new_empty(
            (num_actual_toks, num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        want_lse = bool(self.need_to_return_lse_for_decode)
        kernel_out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.kernel_qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=topk_lengths,
            max_seq_len=sparse_topk_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_topk_capacity,
            kv_scale_format=self.kv_scale_format,
            return_lse=want_lse,
        )
        lse: torch.Tensor | None = None
        if want_lse:
            assert isinstance(kernel_out, tuple), "SM120 sparse MLA did not return LSE"
            out, lse = kernel_out
        else:
            out = kernel_out
        out = out.view(-1, out.shape[-2], out.shape[-1])
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        if lse is not None:
            lse = self._normalize_lse(lse, out.shape[0], out.shape[1])
            lse = lse.masked_fill(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if self.rope_pad:
            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
