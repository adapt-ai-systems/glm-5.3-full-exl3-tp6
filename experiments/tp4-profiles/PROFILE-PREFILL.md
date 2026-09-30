# GLM 5.3 full TP4 prefill — torch profiler breakdown (2026-09-29)

Stack = the kept TP4 stack S4 (g32 + dual-HCA + D3 + atomic), booted with
`--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/prof","torch_profiler_with_stack":false,"ignore_frontend":true}'`
and no CUPTI injection (the torch profiler needs CUPTI). Boot probe pf8K 772 = same as normal serving.
One cold fresh-seed prompt per capture (start_profile -> 1 request, max_tokens 1 -> stop_profile):

| capture | prompt tok | TTFT | tok/s (profiled) | chunks (MNBT 4096) | GPU kernel window |
|---|---|---|---|---|---|
| 8K | 8168 | 10.55 s | 774.6 | 2 (5.16 s + 5.00 s) | 10.37 s |
| 32K | 32451 | 42.19 s | 769.1 | 8 (4.94-5.24 s each) | 41.97 s |

Profiler overhead is in the noise (774.6 profiled vs 772 probe / 783 RESULTS S4).
Traces (all 4 ranks) and the per-kernel-category tables are not included (size); numbers below are from rank 0 unless noted.

## Headline facts

1. **GPU-bound, one stream, no overlap.** Kernels cover 99.4% of the window on every rank. All work (compute and NCCL)
   runs on a single CUDA stream, with 0.0% kernel overlap. The host is far ahead of the GPU ("Command Buffer Full" is 84% of CPU time), so there is no launch or
   Python overhead to win. Idle gaps are 0.6%.
2. **Prefill speed is a fixed per-chunk cost.** Each 4096-token chunk takes ~5.1 s at 8K and ~5.2 s at 32K. Sparse attention
   (index_topk 2048) caps attention work per query, so context length barely matters: sparse_mla_prefill goes 8.2 -> 8.9 ms/layer and the indexer
   goes 1.5% -> 2.4%. That is why pf8K ≈ pf32K. A bigger MNBT cannot help: the GPU is already saturated, and S5 showed E3 caps capacity at 4096.
3. **Ranks are symmetric** (rank0-3 category shares within ±0.4 pt), so rank0 stands for all ranks.

## Breakdown by category (rank0, % of GPU kernel time)

| category | 8K ms | 8K % | 32K ms | 32K % |
|---|---|---|---|---|
| MoE experts: E3 gateup + down + gather (+ fused norm) | 3342 | 32.4 | 13228 | 31.7 |
| **Comms: NCCL all-reduce + all-gather + reduce-scatter** | **2669** | **25.9** | **10512** | **25.2** |
| attention: sparse_mla_prefill + DCP merge (`_correct_attn_cp_out`) | 1491 | 14.5 | 6401 | 15.4 |
| norm / rope / elementwise / copies (triton + aten) | 1454 | 14.1 | 5783 | 13.9 |
| dense GEMMs (FlashInfer MXFP8/W8A8, bf16 MLA bmm, act-quant, marlin->mxfp8) | 1190 | 11.5 | 4728 | 11.3 |
| indexer (DSA fp8 mqa_logits, top-k, pack) | 153 | 1.5 | 1000 | 2.4 |
| idle gaps (window − busy) | 65 | 0.6 | 301 | 0.7 |

Where the anonymous copies come from (CPU parent op via launch correlation, 8K):
`reduce_scatter` contiguous clone 292 ms + `all_gather` reshape clone 252 ms = **544 ms (5.3%) of layout copies around
collectives**. Other items: MLA `masked_fill_` 190 ms, MLA `cat` 122 ms, MLA bf16 `bmm` 208 ms, MoE `to()` copy 99 ms, MoE `zero_` fill 75 ms.
Counting those collective copies, **communication costs ~31% of prefill**.

Where the collectives are (per 4096-token chunk, 78 layers):
- 2 all-reduces per layer (layer outputs; ~157/chunk, median 3.6 ms each). Size = 4096 × 6144 × bf16 = 50.3 MB, which gives
  ~14 GB/s algbw / ~21 GB/s ring busbw, well under the two 200G rails (~50 GB/s).
- 2 all-gathers + 1 reduce-scatter per layer inside MLA attention (DCP2 AG/RS), each ~3.7 ms (~860 ms/chunk). The indexer adds 21 all-gathers per chunk.
- **Every NCCL kernel is `*_RING_LL`**: the LL protocol at 50 MB messages. The env sets no `NCCL_PROTO`, so NCCL 2.30.7 picks
  LL itself. Earlier campaigns measured that *forcing* LL cut prefill 24–35% (glm-tp6-advisor exp16, tp6-speed
  NIGHT-REPORT). The D3 RoCE one-shot path (≤256 KB AR, ≤2 MB AG) never fires in prefill (3 ms total). All large messages go through NCCL.

## Top kernels (8K, rank0; 32K shares are the same within ±1 pt)

| kernel | category | ms | % | calls | avg µs |
|---|---|---|---|---|---|
| `sparse_mla_prefill_mg_kernel<ModelType2,…,32,2048,64,2>` | attention | 1296 | 12.6 | 158 | 8204 |
| `glm6_e3_down` | MoE | 1294 | 12.6 | 152 | 8513 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | comm | 1237 | 12.0 | 320 | 3865 |
| `glm6_e3_gateup` | MoE | 1192 | 11.6 | 152 | 7842 |
| `ncclDevKernel_AllGather_RING_LL` | comm | 849 | 8.2 | 378 | 2246 |
| `glm6_e3_gather` | MoE | 676 | 6.6 | 152 | 4450 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | comm | 583 | 5.7 | 164 | 3556 |
| `at::native::elementwise_kernel` (copies: RS/AG clones, masked_fill) | copy | 495 | 4.8 | 474 | 1043 |
| FlashInfer `DeviceGemmMxfp8GemmSm120` (D2d dense) | GEMM | 484 | 4.7 | 530 | 914 |
| `_correct_attn_cp_out_kernel` (DCP merge) | attention | 195 | 1.9 | 164 | 1188 |
| `at::native::elementwise_kernel` (2nd variant) | copy | 192 | 1.9 | 322 | 598 |
| `tensorrt_llm quantize_with_block_size` (mxfp8 act-quant) | GEMM | 178 | 1.7 | 836 | 213 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 170 | 1.7 | 156 | 1092 |
| `triton_red_fused_fused_add_rms_norm_moe_forward_shared_0` | norm (in MoE op) | 167 | 1.6 | 158 | 1059 |
| `glm_d2_marlin_to_mxfp8` | GEMM prep | 127 | 1.2 | 836 | 152 |
| `CatArrayBatchedCopy` (MLA cat) | copy | 123 | 1.2 | 164 | 749 |
| `cutlass_80_wmma bf16 gemm 32x32` (MLA bmm) | GEMM | 121 | 1.2 | 158 | 765 |

The b12x `W4A16MixedTrellis` MoE kernel is 3.2 ms at 8K and 11.1 ms at 32K (≈0.03%). Prefill MoE runs entirely on E3.

## 3 biggest opportunities (rough sizes; not tried; per 8K GPU time 10.3 s = ~790 tok/s)

1. **Large-message collectives: NCCL picks RING_LL for 50 MB.** NCCL is 2.67 s (26%) at ~21 GB/s busbw.
   A Simple-protocol path for the big prefill messages (NCCL_PROTO=Simple / `^LL`, or a tuner/size switch that keeps LL for
   small decode messages) that doubled effective bandwidth would save **~1.0–1.3 s → roughly +10–14% prefill (≈870–900 tok/s)**.
   This is the cheapest test (one env var and one boot) and the largest single item. Unknown: why NCCL auto-selects LL here
   (possibly the fixed 4 channels + MERGE_NICS tuning model). Decode must be re-checked, because it uses the same collectives.
2. **Hide the communication behind compute (single stream today).** Collectives plus their layout copies are ~3.2 s (31%), fully
   serialized. Removing the 544 ms of RS/AG contiguous/reshape clones (write into contiguous buffers) is **~5%** on its own.
   Real overlap (two micro-batches per chunk, or async AR overlapped with the next layer's attention/shared-expert work) could hide
   up to the whole 31%. Realistically **+10–20%** after item 1, but this is engineering (vLLM model runner + DCP attention code).
3. **MoE E3 path: 3.34 s (32%).** `glm6_e3_gather` (676 ms, 6.6%) is pure token permutation, 4.4 ms/layer, all on one stream.
   Fusing the gather into gateup (index-load inside the grouped GEMM), plus the `to()` copy (99 ms) and `zero_` fill (75 ms) in
   `moe_forward_shared`, is **~0.8 s → ~+8%**. The gateup+down GEMMs themselves (2.49 s) run at roughly
   37 TFLOPS effective (4096 tok × 8 experts × 3·6144·2048 · 2 FLOP × 75 layers / 4 ranks ≈ 46 TFLOP per rank per chunk in
   1.24 s; my arithmetic, not measured against a GB10 roofline). Any headroom there needs kernel tuning.

Smaller items: the MLA attention glue (masked_fill 190 + cat 122 + bmm 208 + DCP merge 195 ≈ 0.7 s, ~7%) and the D2d activation
quant + marlin->mxfp8 conversion (305 ms, 3%; `glm_d2_marlin_to_mxfp8` fires 418×/chunk, so if it re-converts weights on every
call, caching them would save ~1.2%. Not checked).
