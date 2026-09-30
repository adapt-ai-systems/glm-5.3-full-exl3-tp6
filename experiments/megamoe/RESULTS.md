# Phase 1 — live EXL3 MoE path (b12x mixed_trellis one-grid), one layer, GB10 worker node

Layer 40, TP4 rank-slice geometry (H=6144, I=512/rank, 256 experts, top-8, 192 K3 + 64 K4), bf16 activations, real `Exl3MoEMethod._prepare_mixed_rank_sliced_weights` -> `_mixed_rank_sliced_runtime` -> `_apply_mixed_rank_sliced` from the live overlay exl3.py, run in image `glm53-six:e3-v2-attn-l2-v1` with the overlay bind-mounted. Env: VLLM_EXL3_PREFILL_CAPACITY=4096, max_num_batched_tokens=4096, TRELLIS_MAX_M=32 (decode plan) / block_m 32 prefill plan, GLM53_MIXED_PREFILL_CHUNK=1024.

## Setup caveats (read these)

- Only 171/256 experts had a fragment on node 5 for layer 40; the other 85 reuse a same-K expert (whole expert incl. its own rotations; substitution map in `out/layer_info.json`). Weights are real, expert identity is not. Weight bytes per call are real either way (prepared copies are distinct memory).
- The fragments carry PER-EXPERT hidden-side rotations (gate suh != up suh, differ across experts), so this bench runs `broadcast_suh=False, broadcast_svh=False`. If the live TP4 checkpoint is shared-H, the live kernel is a broadcast specialization; the serving head's config was not checked. Rotation bytes are ~1% of traffic so it should not matter for bandwidth.
- The layer is a fake `SimpleNamespace` layer fed to the REAL method code (not a vLLM model), one prepared copy of the layer. L2 defeat = 256 MB `zero_()` write before every timed call (outside the timed region); GB10 L2 is far smaller than 256 MB.
- Decode M=1..20 timed as CUDA-graph replay (one graph per M, fresh x/routes copied into static buffers each replay, 48 replays over 24 pre-generated x/route sets). Prefill M=1024/4096 eager (10 flush+call iters over up to 8 sets). Times are cuda-event time around the call only.
- x = randn*0.5 bf16. `router` = the real layer-40 router (sigmoid + e_score_correction_bias, top8, normalize, x2.5) on that x; `distinct` = every (token,k) a distinct expert while M*8<=256 (so unique experts = 8M up to M=32), random normalized weights. Real activations probably route less uniformly than random x (fewer unique experts per M), so `router` rows are an upper-ish bound on unique experts for real traffic and `distinct` is the hard worst case.
- GPU was otherwise idle; the NVFP4 weight download on this box was running (network/disk, not GPU). Clocks at end of bandwidth test: 2444 MHz sm.

## Measured read bandwidth

| test | best GB/s | median GB/s |
|---|---|---|
| 2GB_sum_int64 | 235.0 | 226.9 |
| 2GB_sum_fp16 | 239.7 | 237.3 |
| 2GB_sum_bf16 | 239.5 | 236.9 |
| 2GB_copy_read_only_equiv | 112.2 | 111.0 |
| 2GB_copy_rw_GBps | 224.4 | 221.9 |
| 4GB_sum_int64 | 236.1 | 215.8 |
| 4GB_sum_fp16 | 240.8 | 239.2 |
| 4GB_sum_bf16 | 221.1 | 198.0 |
| 4GB_copy_read_only_equiv | 116.9 | 112.4 |
| 4GB_copy_rw_GBps | 233.9 | 224.7 |
| 4GB_rowmax_int32 | 243.2 | 240.8 |

Roofline used: **243.2 GB/s** (best pure-read reduction; copy_ rw = 224-234 GB/s combined). Spec 273 GB/s is not achievable; ~89% of spec is what a plain reduction gets.

## Accuracy: b12x mixed_trellis vs overlay per-expert `exl3_gemm` reference

Reference = overlay `Exl3MoEMethod.apply` generic loop (fp16 x, `_exl3_gemm` for w1/w3/w2 per expert, silu*up, weighted index_add fp32). `bf16 floor` = rel L2 of merely rounding the fp32 reference to bf16 (the b12x output is bf16), so err at floor = agreement to output precision.

| M | routing | unique experts | max abs err | ref absmax | rel L2 | bf16 floor rel L2 |
|---|---|---|---|---|---|---|
| 1 | router | 8 | 9.16e-04 | 0.284 | 1.94e-03 | 1.67e-03 |
| 1 | distinct | 8 | 1.01e-03 | 0.300 | 1.96e-03 | 1.68e-03 |
| 5 | router | 35 | 1.06e-03 | 0.327 | 1.93e-03 | 1.65e-03 |
| 5 | distinct | 40 | 1.10e-03 | 0.426 | 1.95e-03 | 1.64e-03 |
| 20 | router | 108 | 1.18e-03 | 0.378 | 1.94e-03 | 1.66e-03 |
| 20 | distinct | 160 | 1.19e-03 | 0.384 | 1.94e-03 | 1.66e-03 |
| 1024 | router | 218 | 1.25e-03 | 0.460 | 1.94e-03 | 1.66e-03 |
| 1024 | distinct | 256 | 1.34e-03 | 0.490 | 1.94e-03 | 1.66e-03 |
| 4096 | router | 231 | 1.28e-03 | 0.486 | 1.94e-03 | 1.66e-03 |
| 4096 | distinct | 256 | 1.96e-03 | 0.645 | 1.94e-03 | 1.66e-03 |

## Timing per call

weight MB = trellis + rotations of the unique experts touched (K3 slice 3.54 MB, K4 4.72 MB, +~40 KB rotations); floor = bytes / 243.2 GB/s.

| M | routing | mode | us median | us min | us p90 | unique experts | weight MB | GB/s (median) | % of read BW | floor us | overhead us (med - floor) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | router | graph | 228 | 218 | 240 | 8.0 | 30.4 | 133 | 55% | 125 | 103 |
| 1 | distinct | graph | 231 | 218 | 242 | 8.0 | 31.1 | 135 | 55% | 128 | 103 |
| 2 | router | graph | 355 | 329 | 379 | 15.6 | 58.5 | 165 | 68% | 240 | 114 |
| 2 | distinct | graph | 371 | 348 | 383 | 16.0 | 62.3 | 168 | 69% | 256 | 115 |
| 3 | router | graph | 482 | 447 | 502 | 22.8 | 86.5 | 179 | 74% | 356 | 126 |
| 3 | distinct | graph | 507 | 484 | 528 | 24.0 | 93.3 | 184 | 76% | 384 | 123 |
| 4 | router | graph | 595 | 536 | 647 | 29.7 | 111.7 | 188 | 77% | 459 | 136 |
| 4 | distinct | graph | 651 | 620 | 672 | 32.0 | 123.9 | 190 | 78% | 509 | 142 |
| 5 | router | graph | 691 | 628 | 752 | 35.7 | 133.7 | 193 | 80% | 550 | 142 |
| 5 | distinct | graph | 791 | 765 | 809 | 40.0 | 155.2 | 196 | 81% | 638 | 152 |
| 6 | router | graph | 821 | 740 | 878 | 41.8 | 157.8 | 192 | 79% | 649 | 172 |
| 6 | distinct | graph | 921 | 891 | 959 | 48.0 | 185.5 | 201 | 83% | 763 | 158 |
| 7 | router | graph | 881 | 798 | 943 | 46.8 | 175.8 | 199 | 82% | 723 | 159 |
| 7 | distinct | graph | 1066 | 1020 | 1094 | 56.0 | 216.5 | 203 | 84% | 890 | 175 |
| 8 | router | graph | 992 | 932 | 1068 | 53.4 | 201.5 | 203 | 84% | 829 | 164 |
| 8 | distinct | graph | 1204 | 1169 | 1327 | 64.0 | 247.0 | 205 | 84% | 1016 | 188 |
| 9 | router | graph | 1103 | 991 | 1199 | 59.4 | 223.9 | 203 | 84% | 921 | 182 |
| 9 | distinct | graph | 1345 | 1287 | 1540 | 72.0 | 277.7 | 207 | 85% | 1142 | 203 |
| 10 | router | graph | 1173 | 1082 | 1308 | 63.7 | 239.0 | 204 | 84% | 983 | 190 |
| 10 | distinct | graph | 1491 | 1444 | 1609 | 80.0 | 310.1 | 208 | 86% | 1275 | 216 |
| 11 | router | graph | 1270 | 1078 | 1419 | 68.8 | 259.0 | 204 | 84% | 1065 | 205 |
| 11 | distinct | graph | 1623 | 1585 | 1678 | 88.0 | 342.2 | 211 | 87% | 1407 | 216 |
| 12 | router | graph | 1357 | 1267 | 1430 | 72.6 | 274.8 | 203 | 83% | 1130 | 227 |
| 12 | distinct | graph | 1778 | 1728 | 1866 | 96.0 | 373.0 | 210 | 86% | 1534 | 245 |
| 13 | router | graph | 1426 | 1226 | 1545 | 77.4 | 291.4 | 204 | 84% | 1198 | 227 |
| 13 | distinct | graph | 1897 | 1848 | 2042 | 104.0 | 402.5 | 212 | 87% | 1655 | 242 |
| 14 | router | graph | 1486 | 1329 | 1634 | 81.8 | 308.1 | 207 | 85% | 1267 | 219 |
| 14 | distinct | graph | 2033 | 1992 | 2111 | 112.0 | 434.6 | 214 | 88% | 1787 | 246 |
| 15 | router | graph | 1586 | 1478 | 1764 | 85.1 | 320.7 | 202 | 83% | 1319 | 268 |
| 15 | distinct | graph | 2176 | 2127 | 2322 | 120.0 | 465.5 | 214 | 88% | 1914 | 262 |
| 16 | router | graph | 1585 | 1458 | 1724 | 88.8 | 334.7 | 211 | 87% | 1376 | 209 |
| 16 | distinct | graph | 2325 | 2283 | 2491 | 128.0 | 496.7 | 214 | 88% | 2042 | 282 |
| 17 | router | graph | 1673 | 1496 | 1784 | 92.1 | 348.2 | 208 | 86% | 1432 | 241 |
| 17 | distinct | graph | 2530 | 2453 | 2744 | 136.0 | 526.2 | 208 | 86% | 2164 | 366 |
| 18 | router | graph | 1723 | 1526 | 1919 | 96.2 | 362.7 | 210 | 87% | 1491 | 232 |
| 18 | distinct | graph | 2597 | 2540 | 2703 | 144.0 | 558.6 | 215 | 88% | 2297 | 300 |
| 19 | router | graph | 1823 | 1599 | 1912 | 99.7 | 376.7 | 207 | 85% | 1549 | 275 |
| 19 | distinct | graph | 2735 | 2685 | 2884 | 152.0 | 590.1 | 216 | 89% | 2427 | 308 |
| 20 | router | graph | 1879 | 1710 | 2013 | 102.4 | 387.1 | 206 | 85% | 1592 | 287 |
| 20 | distinct | graph | 2854 | 2804 | 3006 | 160.0 | 619.4 | 217 | 89% | 2547 | 307 |
| 5 | router | eager | 705 | 639 | 751 | 35.8 | 134.5 | 191 | 78% | 553 | 152 |
| 20 | router | eager | 1851 | 1677 | 1978 | 101.2 | 383.1 | 207 | 85% | 1575 | 275 |
| 1024 | router | eager | 10823 | 10263 | 11052 | 223.6 | 857.2 | 79 | 33% | 3525 | 7298 |
| 1024 | distinct | eager | 11179 | 10766 | 11261 | 256.0 | 991.7 | 89 | 36% | 4078 | 7101 |
| 4096 | router | eager | 31701 | 30925 | 32026 | 233.2 | 897.0 | 28 | 12% | 3688 | 28013 |
| 4096 | distinct | eager | 32195 | 31354 | 32590 | 256.0 | 991.7 | 31 | 13% | 4078 | 28118 |

### Linear fit of decode graph time vs bytes (M=1..20)

- router: t = 84 us + 4.58 us/MB  (marginal 218 GB/s = 90% of read BW)
- distinct: t = 91 us + 4.50 us/MB  (marginal 222 GB/s = 91% of read BW)

Prefill compute check: M=1024 = 8192 (token,expert) rows x 3 x 2 x 512 x 6144 = 155 GFLOP in 10.8 ms = 14 TFLOP/s; M=4096 = 620 GFLOP in 31.7 ms = 20 TFLOP/s. Prefill is compute/dequant-bound, not weight-read-bound, so % of read BW is not the right yardstick there.

## Kernel breakdown (torch profiler, 10 flush+call iterations)

Per MoE call, kernels launched: route pack (2 tiny kernels), one `W4A16MixedTrellisKernel` grid (FC1+act+FC2), `W4A16TopKSumKernel`, one bf16 dtype-copy. No other torch glue (rotations run inside the one grid).

| case | wall span first->last kernel | route pack (prefix+sort) | main mixed_trellis grid | topk_sum | bf16 copy |
|---|---|---|---|---|---|
| M5_eager | 918 us | 4.1 us | 664 us | 4.5 us | 1.0 us |
| M5_graph | 671 us | 3.6 us | 703 us | 4.4 us | 1.1 us |
| M20_eager | 2201 us | 5.0 us | 1718 us | 15.0 us | 9.1 us |
| M20_graph | 2008 us | 3.7 us | 1715 us | 13.6 us | 1.3 us |

(Profiler runs a bit slower than the event timing above; eager first kernel starts ~250 us after call entry = CPU launch latency, hidden in graph mode.)


# Phase 2 — prefill: tensor-core ceiling, block_m sweep, live E3 path, ncu

Same fake layer-40 harness and L2-flush method as Phase 1 (`bench_layer.py`). Prefill is eager. E3 = Mia's live `TP4_E3_PREFILL` path (gather + gateup + down kernels), run from its package on one worker node (cubin rebuilt with `nvcc -arch=sm_121a`, so not byte-identical to the manifest cubin); measured, not copied.

## A. Dense tensor-core peak on one worker node (torch `mm`, best of N)

| op | M | N | K | dtype | best TFLOP/s | median TFLOP/s |
|---|---|---|---|---|---|---|
| mm | 8192 | 8192 | 8192 | float16 | 100.4 | 96.7 |
| mm | 128 | 1024 | 6144 | float16 | 41.9 | 40.5 |
| mm | 128 | 6144 | 512 | float16 | 45.9 | 43.8 |
| mm | 256 | 1024 | 6144 | float16 | 55.6 | 54.5 |
| mm | 256 | 6144 | 512 | float16 | 63.6 | 58.5 |
| mm | 512 | 1024 | 6144 | float16 | 83.0 | 78.7 |
| mm | 512 | 6144 | 512 | float16 | 71.1 | 68.5 |
| mm | 1024 | 1024 | 6144 | float16 | 86.2 | 85.1 |
| mm | 1024 | 6144 | 512 | float16 | 61.7 | 59.6 |
| mm | 4096 | 1024 | 6144 | float16 | 92.5 | 91.5 |
| mm | 4096 | 6144 | 512 | float16 | 63.7 | 62.9 |
| mm | 8192 | 8192 | 8192 | bfloat16 | 100.0 | 97.7 |
| mm | 128 | 1024 | 6144 | bfloat16 | 42.6 | 40.7 |
| mm | 128 | 6144 | 512 | bfloat16 | 46.9 | 44.1 |
| mm | 256 | 1024 | 6144 | bfloat16 | 55.6 | 54.5 |
| mm | 256 | 6144 | 512 | bfloat16 | 62.6 | 60.4 |
| mm | 512 | 1024 | 6144 | bfloat16 | 85.8 | 78.6 |
| mm | 512 | 6144 | 512 | bfloat16 | 72.0 | 71.4 |
| mm | 1024 | 1024 | 6144 | bfloat16 | 86.9 | 85.8 |
| mm | 1024 | 6144 | 512 | bfloat16 | 61.5 | 60.2 |
| mm | 4096 | 1024 | 6144 | bfloat16 | 93.1 | 91.5 |
| mm | 4096 | 6144 | 512 | bfloat16 | 63.7 | 62.6 |
| scaled_mm | 8192 | 8192 | 8192 | e4m3 | 213.5 | 196.7 |

fp16/bf16 dense peak ~100 TFLOP/s. The two per-expert GEMM shapes (gate/up: N=1024,K=6144 ; down: N=6144,K=512) reach ~93 / ~64 TFLOP/s at M=4096 and 42-83 at M=128-512. e4m3 `_scaled_mm` is 213 TFLOP/s (2x).

## B. b12x prefill `block_m` sweep (one-grid mixed_trellis, router routing)

| M | block_m | us median | main grid us | route+topk us | rel L2 |
|---|---|---|---|---|---|
| 1024 | 32 | 10407 | 9937 | 595 | 1.94e-03 |
| 2048 | 32 | 17515 | 15493 | 1250 | 1.94e-03 |
| 4096 | 32 | 31690 | 28519 | 2484 | 1.94e-03 |
| 1024 | 64 | 12426 | 11655 | 611 | 1.94e-03 |
| 2048 | 64 | 18937 | 17145 | 1187 | 1.94e-03 |
| 4096 | 64 | 32751 | 29694 | 2389 | 1.94e-03 |
| 1024 | 128 | FAILED: `ValueError('FC2 route subtile must be an allowed divisor of the packed route block: packed` | | | |
| 2048 | 128 | FAILED: `ValueError('FC2 route subtile must be an allowed divisor of the packed route block: packed` | | | |
| 4096 | 128 | FAILED: `ValueError('FC2 route subtile must be an allowed divisor of the packed route block: packed` | | | |

block_m=32 is fastest at every M; 64 is 5-20% slower; 128 is rejected at planning by the FC2 subtile check. Accuracy identical for 32/64. Main grid is ~28.5 of 31.7 ms at M=4096.

## C. E3 (live prefill path) vs b12x prefill

Per-kernel us are torch-profiler GPU times per call; `fill/memset` row in `out/e3.jsonl` also contains the 256 MB L2-flush fill (~1.3-1.9 ms) and is excluded here.

| M | E3 wall us | b12x wall us (block_m 32) | E3 speedup | gather us | gateup us | down us | glue+zero+cast us (rest) | rel L2 vs ref | rel L2 E3 vs b12x |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3542 | n/a | n/a | 107 | 2383 | 1052 | 97 | 1.94e-03 | 1.01e-03 |
| 1024 | 8194 | 10407 | 1.27x | 1105 | 3951 | 2635 | 261 | 1.94e-03 | 1.01e-03 |
| 2048 | 12667 | 17515 | 1.38x | 2288 | 5071 | 4540 | 482 | 1.94e-03 | 1.01e-03 |
| 4096 | 22389 | 31690 | 1.42x | 4412 | 7842 | 8510 | 873 | 1.94e-03 | 1.01e-03 |

At M=4096 E3 is 22.4 ms vs 31.7 ms (1.42x, 29% less time). Its 3 big kernels are 4.4 + 7.8 + 8.5 = 20.8 ms; glue (routing sort/scan, zero_, bf16 cast) is ~0.9 ms. Compute achieved ~30 TFLOP/s (620 GFLOP / 20.8 ms) vs ~64-93 TFLOP/s dense fp16 on the same shapes.

## D. ncu on E3 at M=4096 (`out/ncu_e3_4096.csv`, `out/ncu_e3_full_4096.csv`; view: `python3 ncu_show.py <csv>`)

| kernel | duration | grid | regs/thread | occupancy | issue slots busy | SM busy | shared bank conflicts | cycles / issued instr |
|---|---|---|---|---|---|---|---|---|
| gateup | 7.77 ms | 2048 | 128 | 33% (2 blocks/SM; limit = registers and 48 KB smem) | 45% | 64% | 3.5 M | 9.5 |
| down | 8.45 ms | 12288 | 128 | 33% | 24% | 29% | 20.8 M | 17.9 |
| gather | 4.5 ms | 49152 | 39 | n/a | n/a | n/a | n/a | n/a |

gateup: ALU pipe 24%, fmaheavy 22.5%, LSU 40%, Mem Pipes Busy 39.8%. down: Mem Pipes Busy 21.3%. gather writes 2 x 400 MB fp16 copies (LSU 36.6%).

Not obtainable on GB10 with this ncu: `sm__pipe_tensor_op_hmma_*` and every `dram__*` metric come back `n/a`, and the full set has no tensor-pipe row, so tensor-pipe and DRAM utilisation of E3 are unmeasured. Reading from what is measured: both big kernels are latency/issue-bound (33% occupancy, register + smem limited, 9.5-18 cycles per issued instruction), neither DRAM-bound nor evidently tensor-bound. `down` is worse than `gateup` (24% vs 45% issue slots, 6x the bank conflicts). M=5 b12x ncu was not run (optional).


## E. Stage 1: `exl3mm_fc2` + `exl3mm_finalize` (drop-in for E3 down + `out.zero_()` + bf16 cast)

Sources: `kernels/exl3mm_fc2.cu`, `kernels/exl3mm.py`; harness `stage1_bench.py` (`out/stage1.jsonl`), `time_fc2.py`, `wrbw.py` (`out/wrbw.json`), ncu `out/ncu_s1v2_*.csv`. one GB10 node, GLM 5.3 layer 40 experts, L2 flushed before every timed call.

### E.1 Design (v2, commit 2f3d1ab)
- fc2: one CTA of 16 warps (512 threads), 1 CTA/SM, split into two 8-warp ping-pong groups so one group's epilogue overlaps the other's main loop. Group 1 starts after `__nanosleep(STAGGER_NS)` (default 2000; 0 measures the same).
- 64-row A tile (intermediate, fp16, 512 K) stays resident in shared memory, XOR-swizzled. Each group works a 128-column tile, one warp per 16-column block. TR3 trellis words come through a 4-stage cp.async ring and are decoded in registers (3INST codebook, `LaneDec`) straight into the mma fragments.
- Epilogue per group: fp32 staging (stride 136) -> round z to fp16 -> Hadamard-128 (4 in-register + 5 shuffle stages) x 0.088388347648 -> x `down_svh` -> x route weight -> fp16 store to `ybuf[R, 6144]`.
- `exl3mm_finalize`: one CTA per token, sums its 8 route rows in fixed k order in fp32, writes bf16 (deterministic, no atomics, no zero_).
- 99,328 B shared memory, 118 registers, 16 barriers, no spills.

### E.2 Timing, fc2 + finalize vs E3 down + zero + cast (harness `stage1_bench.py`, median us)

| M | routing | unique experts | mine fc2+finalize | E3 down+zero+cast | speedup |
|---|---|---|---|---|---|
| 64 | router | 163 | 1049 | 1070 | 1.02x |
| 64 | distinct | 220 | 1465 | 1494 | 1.02x |
| 1024 | router | 266 | 2418 | 2893 | 1.20x |
| 1024 | distinct | 256 | 2607 | 3180 | 1.22x |
| 2048 | router | 387 | 3643 | 5092 | 1.40x |
| 2048 | distinct | 382 | 3675 | 5204 | 1.42x |
| 4096 | router | 646 | 6493 (min 6196) | 9607 | 1.48x |
| 4096 | distinct | 629 | 6190 (min 6135) | 9757 | 1.58x |

At M=4096: fc2 alone 4.35 ms, finalize 2.0 ms. Route prep is not included on either side. Target of < 6 ms total at M=4096 is **not met**: 6.2-6.5 ms (best single runs 6.14-6.2 ms).

### E.3 Sanity checks and the corrected fc2 number
An earlier interim message quoted fc2 at 2.76 ms and fc2+finalize at 4.75 ms. **Those numbers were wrong and are withdrawn.** Checks:

1. Write bandwidth : `zero_`/`fill_` 1 GB best 193-197 GB/s (2 GB: 182-187); device copy read+write 219-225 GB/s combined (best); read-only peak 243 GB/s (Phase 1).
2. Floors at those rates: fc2 moves 327 MB of weights + 403 MB of ybuf writes = 730 MB, i.e. >= 3.8 ms at ~190 GB/s. finalize moves 403 MB + 50 MB = 453 MB, i.e. >= 2.06 ms at ~220 GB/s. Sum floor ~5.9 ms. 2.76 ms for fc2 would need 264 GB/s, above the measured read peak, so it could not have been a real full run.
3. Duration cross-check, ncu `gpu__time_duration` with `--clock-control none`: fc2 4.196 ms, finalize 2.009 ms (total 6.2 ms). cuda-event timing in `time_fc2.py` (L2 flushed): fc2 4.17-4.5 ms, fc2+finalize 6.2-6.5 ms, for `STAGGER_NS`=0 and 2000 alike. The nanosleep is not what changes timing. The stale 2.76 figure is not reproducible on the current sources; the container copy of `time_fc2.py` had been out of date (no POISON path), which I only found and fixed now. I did not identify the exact cause of the earlier reading.
4. NaN poison: `ybuf` filled with NaN before each timed fc2 call, all valid route rows finite after it (`poison_all_finite: true`) at 4.2-4.4 ms for `STAGGER_NS`=0 and 2000. (A `-DSTAGGER_NS=-1` variant sleeps ~4 s per warp start and runs at 14 ms; ignore it.)

Verdict: fc2 is at ~4.2 ms vs a ~3.8 ms write+read floor (~90%); finalize is at its floor (2.0 ms vs 2.06 ms). The round trip through a 403 MB fp16 `ybuf` sets the ~5.9 ms lower bound of this two-kernel design; getting materially under 6 ms needs removing that round trip (Stage 2 fusion), not more tuning of Stage 1.

### E.4 Accuracy (`out/stage1.jsonl`; rel L2 vs `ref_moe`, and vs E3 output)

| M | routing | mine vs ref | E3 vs ref | mine vs E3 | max abs vs E3 |
|---|---|---|---|---|---|
| 64 | router | 1.95e-3 | 1.94e-3 | 8.9e-4 | 0.0020 |
| 64 | distinct | 1.96e-3 | 1.95e-3 | 9.0e-4 | 0.0020 |
| 1024 | router | 1.95e-3 | 1.94e-3 | 8.8e-4 | 0.0020 |
| 1024 | distinct | 1.95e-3 | 1.94e-3 | 8.9e-4 | 0.0020 |
| 2048 | router | 1.95e-3 | 1.94e-3 | 8.8e-4 | 0.0020 |
| 2048 | distinct | 1.95e-3 | 1.94e-3 | 8.9e-4 | 0.0020 |
| 4096 | router | 1.95e-3 | 1.94e-3 | 8.8e-4 | 0.0020 |
| 4096 | distinct | 1.95e-3 | 1.94e-3 | 8.9e-4 | 0.0039 |

Output is bit-deterministic across repeats and route tables equal E3's at every M. No NaNs.

### E.5 ncu, v2 (M=4096, clocks locked; `out/ncu_s1v2_fc2.csv`, `out/ncu_s1v2_fin.csv`)

| kernel | duration (locked clocks) | regs | occupancy achieved (theoretical) | issue slots busy | cycles / issued instr |
|---|---|---|---|---|---|
| fc2 | 4.83 ms | 118 | 30% (33%; 1 block/SM: registers, 99 KB smem, barriers) | 45.8% | 9.2 |
| finalize | 1.99 ms | 40 | 93% | 2.7% | n/a |

v1 (single group, `out/ncu_s1v1_*`) was 4.56 ms fc2 at M=4096; v1 ablation: no epilogue 3.32 ms, no mma 4.07 ms, no dequant 4.33 ms.

## F. Decode: `VLLM_EXL3_TRELLIS_MAX_M` knob and fixed cost (`run_decode_knob.py`, `out/knob_{32,20,8}.log`)

CUDA-graph replay of `_apply_mixed_rank_sliced` (live b12x decode plan), layer 40, L2 flushed (256 MB write) before every replay, 72 replays per point, median us (min / p90 in the logs). `router` = real router routing (24 rotating sets); `same8` = every token routed onto the same 8 experts (8 unique experts total), so bytes are constant in M.

| M | routing | unique experts | MAX_M=32 (live) | MAX_M=20 | MAX_M=8 |
|---|---|---|---|---|---|
| 1 | router | 8.0 | 231.9 | 230.1 | 232.4 |
| 1 | same8 | 8.0 | 234.2 | 233.1 | 232.1 |
| 5 | router | 35.6 | 713.0 | 733.6 | 697.5 |
| 5 | same8 | 8.0 | 238.4 | 233.2 | 235.8 |
| 10 | router | 63.3 | 1214.3 | 1194.3 | n/a (M>8) |
| 10 | same8 | 8.0 | 255.3 | 251.3 | n/a |
| 20 | router | 101.8 | 1936.1 | 1854.2 | n/a |
| 20 | same8 | 8.0 | 295.2 | 299.9 | n/a |

- **The knob changes nothing.** Differences between 32/20/8 are within run-to-run noise (spread of medians +-2-3%, p90 spread larger than the differences). MAX_M only sets the decode plan capacity; the decode plan is the same for M <= 8. (MAX_M=8 with M>8 falls to the prefill plan; not timed, as instructed.)
- **Fixed cost vs bytes.** Weight bytes are ~3.75-3.79 MB per expert (trellis + rotations), i.e. 15.6 us per expert at the measured 243 GB/s read peak. Linear fit over router runs (MAX_M=32): us ~= 91 + 17.6 x unique_experts (points 8/35.6/63.3/101.8 experts -> 232/713/1214/1936; slope from consecutive points 17.4-18.2 us/expert = ~89% of read peak). The ~91 us intercept is the fixed cost per call and matches the earlier ~90 us fit.
- `same8` confirms it: at M=1 and M=5 with 8 experts (floor 125 us) the call takes 232-238 us, i.e. ~107 us above the bytes floor, independent of M; the M-dependent part is small (M=10: +20 us, M=20: +60 us over M=1, from more rows per expert).
- So at M<=5 the fixed ~90-110 us is what is left after bytes and is not tunable through this knob.
