# EXL3 megamoe — report (2026-09-29/30)

Question: does a megamoe-style fused MoE for the EXL3 TR3 experts speed up GLM 5.3 full TP4?
Evidence: RESULTS.md (one real layer, layer 40, on one worker node) and the TP4 torch profiles
(../tp4-profiles/PROFILE-PREFILL.md, PROFILE-DECODE.md).

## 1. How much the experts take, and why

| phase | routed-expert kernels | share | bound by |
|---|---|---|---|
| decode (MTP4, M=5 steps) | b12x `W4A16MixedTrellisKernel` + TopKSum | **40.0%** of decode kernel time (845 ms / 2113) | weight reads: 80-85% of the node's read BW (243 GB/s) at M>=5, plus a **~90 us fixed cost per call** (fit t = 84-91 us + 4.5 us/MB) |
| prefill (4096-token chunks) | E3 `glm6_e3_gather` 4.45 + `gateup` 7.84 + `down` 8.51 ms/layer | **32.4%** of prefill kernel time | not DRAM or tensor peak: ~30 TFLOP/s vs 93 (gate/up) / 64 (down) dense peak; ncu = 33% occupancy, issue/latency bound; gather writes 2x400 MB rotated copies; down does fp32 float4 atomics |

The live decode path is already megamoe-shaped (route pack -> one cooperative grid FC1/act/FC2 -> top-k sum,
3 launches, no torch glue under graphs), so "fusing" gives nothing there; only bandwidth efficiency can.

## 2. Prototype: `exl3mm_fc2` + `exl3mm_finalize` (prefill down projection, replaces E3 down + zero + cast)

Our code (kernels/exl3mm_fc2.cu, exl3mm.py; skeleton ideas from Matt Mastracci's megamoe `moe_prefill.cu`, kindlingai/glm-5.3-flash-gx10; no code copied). Reads the live
prepared trellis buffers zero-copy, decodes K3/K4 mcg trellis straight into mma fragments, output Hadamard +
svh in the epilogue, per-route fp16 rows + deterministic finalize (no atomics). 16 warps as two ping-pong groups
over a resident 64x512 A tile.

Correctness on real layer-40 weights: rel L2 vs the overlay's per-expert exl3_gemm reference 1.95e-3
(E3 1.94e-3, b12x 1.94e-3; bf16 rounding floor 1.66e-3); vs E3 output 8.8-9.0e-4, max abs 0.002 (0.0039 worst);
no NaN; deterministic; M = 64/1024/2048/4096, router + distinct routing.

Speed (median us, router routing, L2 flushed; ncu-confirmed fc2 4.20 + finalize 2.01 ms at M=4096):

| M | ours fc2+finalize | E3 down+zero+cast | speedup |
|---|---|---|---|
| 64 | 1049 | 1070 | 1.02x |
| 1024 | 2418 | 2893 | 1.20x |
| 2048 | 3643 | 5092 | 1.40x |
| 4096 | 6493 | 9607 | **1.48x** (distinct 1.58x) |

It sits at ~90% of its own DRAM floor (327 MB weights + 403 MB fp16 route rows written, then read back:
floor ~5.9 ms). Going lower needs the route-row round trip gone (a fused FC1->FC2 or a different combine).

Decode token counts 1-20: no new decode kernel was built — the live b12x decode kernel was measured instead
(RESULTS.md phase 1: M=1 228 us ... M=5 691 us ... M=20 1879 us per layer, 55% -> 85% of read BW).

## 3. End-to-end estimate

- **Prefill, stage 1 as built:** -3.1 ms per MoE layer per 4096 chunk x 76 MoE calls/chunk = -0.24 s of a
  ~5.1 s chunk -> **~+5% prefill (783 -> ~820 tok/s pf8K)**. Not booted; estimate from one-layer timing.
- **Prefill, stage 2 (fused gather+rotate+gate/up, not built):** removes the 4.4 ms gather + 800 MB traffic,
  but each CTA sharing a segment must redo the per-expert had128(x*suh) rotation (4 N-tiles) on a kernel that is
  already issue-bound, and smem is ~full at TILE_M=64. My estimate: roughly break-even to +3-5% more prefill;
  a real experiment, not a sure win.
- **Decode:** perfect kernel = 0 fixed cost + 100% read BW -> saves ~108 us/call x 75 = ~8 ms of a ~96 ms round
  -> **+9% decode ceiling**, realistic +3-5%. Not built.
- Decode fixed cost: VLLM_EXL3_TRELLIS_MAX_M 32/20/8 changes nothing (+-3%, RESULTS.md F). With routes forced
  onto 8 experts, M=1..5 costs 232-238 us vs a 125 us byte floor: ~107 us is fixed inside the cooperative grid
  (rotation phase / grid barriers / ramp), independent of M. Only a kernel change can reach it.
- NCCL RING_LL note from the profile: already tested (NCCL_PROTO=Simple = noise, Simple + 16 channels = -5%; RESULTS.md TP4 rows S6/S6b). Not a lever.

## What was not done
- No live-server swap or boot in this phase. Serving integration = a hook that swaps E3's down launch
  for exl3mm_fc2+finalize (E3 gather/gateup unchanged); needs a boot + quality gate.
- Stage 2 and any decode kernel not built (reasons above).
- Bench caveat: the bench node holds only 171/256 layer-40 expert fragments (TP6 layout); the other 85 reuse a same-K
  expert. Bytes and shapes are real; expert identity is not.

## TP4 serving result (S7b, 2026-09-29/30) — NOT KEPT
- S7 was invalid (sitecustomize RecursionError; fixed by putting the E3 runtime dir first on PYTHONPATH).
- S7b with the hook confirmed active on all 4 ranks: prefill pf8K 796.4 / pf32K 779.1 vs same-window ctl 778.4 / 764.3
  (+2.3% / +1.9%, reproduced 795.7 / 779.0). Well under the +5% estimate from the per-layer test (21.9 -> 18.3 ms at M=4096).
- Structured decode 41.36 (rebench 41.19) vs ctl 42.93: ~3.5% slower per MTP round at equal rounds. Cause not found
  (hook only replaces rt.apply, reached only at M>32). Keep rule fails -> KL gate not run; best-tp4 restored.
- Campaign rows: RESULTS.md, TP4 rows S7*.
