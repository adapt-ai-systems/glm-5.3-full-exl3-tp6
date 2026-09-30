# GLM 5.3 full TP4 decode — torch profiler snapshot (2026-09-29)

Same profiling boot as PROFILE-PREFILL.md (S4 stack, torch profiler, no CUPTI injection). Capture script (not included):
a fixed ~2K-token prompt (2050 tok) is warmed into the prefix cache, then start_profile -> the same prompt, `enable_thinking` false,
`ignore_eos`, max_tokens 64 -> stop_profile. Wall 2.95 s for 64 tokens. MTP4 FULL graphs, so the decode steps are
`generation_1(5)` (capture size 5).
Traces and the vLLM op table are not included (size).

The request carried a 130-token uncached tail prefill, i.e. 2 `execute_context_1` windows (454 + 402 ms). Those are excluded
below. Decode-only = 2113 ms of kernel time over a 2390 ms window (88% GPU busy).

**Share of the b12x mixed-trellis MoE grid:** `W4A16MixedTrellisKernel` + `W4A16TopKSumKernel` = **845 ms = 40.0% of decode
kernel time** (1662 + 1662 calls). In cold prefill the same kernel is 3.2 ms / 10.3 s at 8K and 11.1 ms / 41.7 s at 32K
(≈0.03%). Prefill MoE runs on E3 (`glm6_e3_*`, 32%).

## Top 25 kernels, decode steps only (rank0, self CUDA time)

| kernel | self CUDA ms | % of decode kernel time | calls |
|---|---|---|---|
| `kernel_cutlass_kernel_b12xmoe_sharedkernelsw4a16mixed_trellisW4A16MixedTrellisKernel_object_at_` | 834.2 | 39.5 | 1662 |
| `void marlin::Marlin<1125899906909960l, 1125899907892224l, 1125899906909960l, 1125899906909960l,` | 201.8 | 9.5 | 4977 |
| `kernel_cutlass_kernel_b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__ptri32gmemalign16_` | 167.0 | 7.9 | 3558 |
| `void marlin::Marlin<1125899906909960l, 2814749767172868l, 1125899906909960l, 2814749767106568l,` | 161.9 | 7.7 | 1930 |
| `void marlin::Marlin<1125899906909960l, 2814749767172868l, 1125899906909960l, 2814749767106568l,` | 140.9 | 6.7 | 2519 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 97.7 | 4.6 | 3981 |
| `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv3_2_kernel<(ModelType)2, 32, 2048, 64>(` | 89.7 | 4.2 | 1725 |
| `void marlin::Marlin<1125899906909960l, 1125899907892224l, 1125899906909960l, 1125899906909960l,` | 51.4 | 2.4 | 1575 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(cut` | 47.8 | 2.3 | 2256 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 41.8 | 2.0 | 1726 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8>(cutl` | 37.1 | 1.8 | 1660 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align8>(cutl` | 34.0 | 1.6 | 1659 |
| `void vllm::topKPerRowDecode<1024, true, false, true>(float const*, int const*, int*, int, int, ` | 14.5 | 0.7 | 528 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloa` | 13.1 | 0.6 | 2190 |
| `kernel_cutlass_kernel_b12xmoe_sharedkernelsw4a16kernelW4A16TopKSumKernel_object_at__tensorptrf1` | 10.6 | 0.5 | 1662 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::dire` | 10.1 | 0.5 | 4980 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIter` | 10.0 | 0.5 | 3855 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(ano` | 6.2 | 0.3 | 3407 |
| `void vllm::concat_and_cache_ds_mla_kernel<__nv_bfloat16, unsigned char, (vllm::Fp8KVCacheDataTy` | 6.0 | 0.3 | 1726 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::inde` | 5.4 | 0.3 | 1945 |
| `void flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_merge_kernel<32, 512, 64, 8>(__nv_bfl` | 5.2 | 0.2 | 1725 |
| `kernel_cutlass_kernel_b12xcommroce_allgather_cute_RoceAllGatherLaunch_object_at__ptri32gmemalig` | 5.1 | 0.2 | 109 |
| `triton_red_fused_fused_add_rms_norm_moe_forward_shared_0` | 5.1 | 0.2 | 1663 |
| `triton_red_fused_fused_add_rms_norm_2` | 4.6 | 0.2 | 1638 |
| `triton_red_fused_4` | 4.5 | 0.2 | 1218 |
