# Measured results

All numbers are single-request (one stream) unless a row says otherwise, measured on our own cluster between
2026-09-26 and 2026-09-30. Unit: tok/s. They are workload-specific benchmark numbers, not a task-quality evaluation,
and most rows are **one boot, one bench run**. Run-to-run spread on the same boot was about ±2% for prefill and ±3–5%
for prose decode (see the repeat rows).

**How the benchmarks work** (`eval/pbench.py`, `eval/pbench_tp4.py`):
- *Prefill 8K / 32K*: prompt tok/s of a cold request with 8,192 / 32,768 prompt tokens (fresh random seed, no prefix hits).
- *Prose decode*: 3 × 512-token story completions at T=0.7, `reasoning_effort=low`, content tokens only, end-to-end.
  Before 2026-09-26 our metric timed reasoning text instead; rows marked † use the old metric and are not comparable.
- *Accept*: MTP draft acceptance rate. *Tok/round*: tokens emitted per decode step.
- *Structured / code* (TP4): median of 5 streamed runs of a frozen structured-output prompt and a code prompt,
  T=0, 512 tokens (prompt harness adapted from Matt Mastracci's decode repro, not included).

## TP6 (6 × DGX Spark)

EXL3 TR3 3.25 bpw re-fragmented six ways, TP6, DCP1, MTP. Cumulative stack; each row adds to the one above it.

| step | stack | pf 8K | pf 32K | prose | accept | note |
|---|---|---|---|---|---|---|
| base† | e3-v2 image, MTP k=4 | 961 | 937 | 20.6† | 35.5% | old decode metric (reasoning text) |
| base | same, `reasoning_effort=low` content-only metric | 952 | 927 | 19.6 | 30.1% | first row on the current metric (D1 prob. drafting + async, both neutral) |
| D3 | + RoCE one-shot all-reduce/all-gather | 955 | 931 | 19.96 | 28.9% | +2%, within noise alone |
| D2 | + FP8 (MXFP8 Marlin W8A16) dense layers | 925 | 892 | 26.5 | 29.1% | +35% decode vs base, −4% prefill |
| MTP k=3 | graphs 4/8/12/16 | 923 | 885 | 28.8 | 37.6% | |
| D2b | large-M rows → dequant + BF16 GEMM | 936 | 921 | 28.1 | 37.1% | prefill mostly recovered |
| MTP k=2 | graphs 3/6/9/12 | 942 | 929 | 29.4 | 47.4% | |
| D2d | large-M rows → MXFP8 W8A8 GEMM (load-time self-check passes) | 999 | 958 | 29.3 | 49.5% | "previous best" (FP8 stack) |
| **D8** | + GPTQ int4 g32 dense decode (M < 256) | 973 | 935 | **32.2** (33.3 / 32.2 / 31.3) | 49.3% | KL gate PASS (below); +10% decode, −2.5% prefill |
| | + Marlin atomic-add, repeat boots | 989–998 | 944–960 | 31.9–33.3 | 48–50% | 3 runs on one boot; atomic-add output drift ~3.5e-3 rel (BF16 rounding level, not bit-exact) |
| **kept** | + fastload + load filter (boot 250 → 161 s) | 1000 | 958 | **33.8** | 51.4% | inference unchanged; one boot |
| | long context, same boot | 64K: 921 | 128K: 854 | 32.0 | 48.4% | TTFT 70.5 s at 64K, 152 s at 128K |

KV pool 803,968 tokens (fp8_ds_mla, 44 GB/rank), max context 360,000 per request.

### MTP k=4 vs k=2 (TP6 kept stack, `eval/accept.py`, T=0, 512 tokens, streamed tok/s)

| workload | k=4 tok/round | k=4 tok/s | k=2 tok/round | k=2 tok/s | k=4 vs k=2 |
|---|---|---|---|---|---|
| structured | 4.77 | 60.6 | 2.97 | 46.6 | +30.1% |
| code | 3.57 | 46.2 | 2.62 | 42.4 | +9.0% |
| prose | 2.78 | 37.6 | 2.32 | 38.2 | −1.7% |

Each extra draft costs ~7.9 ms per round (linear). On the pbench prose metric k=4 measured 27.7 vs 32–34 for k=2; on 40
real single-turn prompts from our own usage (not published) k=4 was 35.1 vs 36.3. Weighted by our own traffic mix
(mostly reasoning and tool calls) we estimated k=4 ~8% faster overall and ship k=4. For prose-heavy use, k=2 is the
better setting (`--speculative-config` with `num_speculative_tokens: 2` and capture sizes 3/6/9/12).

### Concurrency (previous FP8 stack, KV 49 GB/rank, max-num-seqs 8; not re-run on the kept stack)

| concurrent requests | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| per-request tok/s | 33.0 | 25.3 | 18.8 | 15.7 | 12.8 |
| aggregate tok/s | 32.8 | 49.1 | 72.9 | 92.4 | 97.6 |

### TP6 quality gate (`eval/d8_kl.py`)

16 fixed prompts × 256 greedy tokens (4,096 positions, `eval/kl-seqs.jsonl`), top-20 prompt logprobs, compared to a
BF16-dense reference boot (R). All arms run with every row forced through the small-M (decode) path. Rule: KL at most
2× the FP8 dense arm's, and top-1 agreement drop under 1 point.

| arm | KL vs R | top-1 | result |
|---|---|---|---|
| A: FP8 dense (control, defines the limit) | 0.0166 | 97.09% | limit: KL ≤ 0.0332, top-1 ≥ 96.09% |
| RTN int4 g128 | 0.0544 | 94.26% | FAIL |
| GPTQ int4 g64 | 0.0313 | 95.75% | FAIL (top-1) |
| **GPTQ int4 g32 (shipped)** | **0.0284** | **96.51%** | **PASS** |

## TP4 (4 × DGX Spark)

Native TP4 export, DCP2, MTP k=4 (FULL graphs 5/10/15/20), E3 prefill, KV 20 GB/rank → pool 730,750 tokens,
max context 360,000. Every row is one boot. S0 is the control; each kept step adds to the one above.

| step | stack | pf 8K | pf 32K | prose | accept | tok/round | structured | code | kept |
|---|---|---|---|---|---|---|---|---|---|
| S0 | control (fastload) | 648.7 | 642.2 | 14.96 | 0.302 | 2.21 | 32.62 | 25.10 | reference |
| S1 | + D2d FP8/W8A8 dense | 658.0* | 651.8* | 18.76 | 0.302 | 2.21 | 38.43 | 31.97 | yes |
| S2 | + dual-HCA NCCL (both RoCE HCAs, `NCCL_IB_MERGE_NICS=1`) | 777.9 | 763.1 | 18.70 | 0.294 | 2.17 | 39.36 | 31.25 | yes |
| S3 | + D3 RoCE one-shot AR/AG | 782.0 | 765.7 | 19.49 | 0.291 | 2.17 | 40.00 | 32.04 | yes |
| **S4** | + GPTQ int4 g32 dense decode + Marlin atomic-add | **783.2** | **769.4** | **19.78** | 0.284 | 2.14 | **42.63** | **35.15** | **yes (shipped)** |
| S5 | + max-num-batched-tokens 8192 | — | — | — | — | — | — | — | boot fails: the TP4 E3 prefill kernel accepts capacity 2048/4096 only |
| S6 | S4 + `NCCL_PROTO=Simple` | 776.6 | 764.4 | 20.25 | 0.282 | 2.13 | 42.41 | 34.06 | no (noise) |
| S6b | S4 + Simple + 16 channels | 740.1 | 737.4 | — | — | — | — | — | no (−5%) |

\* S1 needed a reboot: the first S1 boot came up slow (pf 8K ~500). Fast-load boots on TP4 are sometimes slow
(2 of 8 good-config boots in this run); the launcher procedure is to probe one 8K prefill after boot and reboot once
below ~580 tok/s. Cause not found.

S4 vs S0: prefill +21% (8K) / +20% (32K), prose +32%, structured +31%, code +40%. S4 boot to healthy: 142 s.

TP4 KL gate (same method and prompts as TP6, TP4's own FP8 arm A: KL 0.01621, top-1 97.12% → limits KL ≤ 0.03242,
top-1 ≥ 96.12%): **GPTQ g32 KL 0.02676, top-1 96.24% → PASS, with a thin top-1 margin (0.12 points)**. The same TP4
solve scored 96.04% (FAIL) on the day-earlier stack without D3/dual-HCA/atomic-add, so treat TP4 quality as borderline
and re-run the gate on your own boot.

Also tried on TP4, not kept (`experiments/megamoe`): an EXL3 fused down-projection kernel for prefill: +2.3% / +1.9%
prefill, but structured decode −3.7% at equal MTP rounds (cause not found).

## TP4 vs TP6

| | TP6 kept | TP4 kept |
|---|---|---|
| nodes | 6 | 4 |
| pf 8K / 32K | 1000 / 958 | 783 / 769 |
| prose (pbench) | 33.8 (k=4 boots: 27.7) | 19.8 (k=4) |
| KV pool | 803,968 | 730,750 |
| boot to healthy | ~160 s | ~142 s |

The TP6 prose number is from a k=2 boot; TP6 with k=4 (what `launch/tp6/launch.sh` starts) measured 27.7 on pbench
prose. The structured/code harnesses differ between the TP6 and TP4 tables, so we do not compare those columns.

## Tried and dropped (TP6 unless noted)

| idea | result |
|---|---|
| MoE decode tile / cp.async stage tuning | ≤ ±5% on one layer, no-go: the MoE kernels already read at 80–86% of the ~231 GB/s GB10 ceiling |
| DSpark drafter (K=2/3/4) instead of MTP | prose 25.5–28.6, below MTP k=2 |
| FP8 or BF16 precision for the MTP layer | no acceptance gain; acceptance is capped by the MTP head, not its precision |
| lossy lenient acceptance | +10% prose, lossy; rejected |
| RTN int4 dense | fails the KL gate |
| max-num-batched-tokens 8192 | +2% prefill, not adopted while the hang (below) is open |
| jumbo frames (MTU 9000) | half-jumbo cost 5% prefill; full-jumbo runs hit the known hang, so jumbo is unmeasured, not proven harmful |
| 8 NCCL channels, CUDA graph size 1, static expert placement | no gain |
| small-M kernel rewrites (router, MLA bmm, shared down) | ~1% estimated, parked |
| adaptive / phase-aware draft length (`experiments/d13-adaptive-k`) | +12–16% pbench prose and +5–7% on real prompts vs fixed k=4, tool calls unchanged; partial A/B only, not shipped |

## Known problems

- **Prefill hang (TP6, not root-caused).** Mostly the first cold ≥8K prefill right after a short decode, often the
  first request after boot. Symptom: all ranks' GPUs busy at low power, no RoCE traffic, a GPU kernel never completes;
  seen with and without D3, at MTU 1500 and 9000, at max-num-batched-tokens 4096 and 8192. On our cluster it hung in
  5 of 9 bench runs without a CUPTI injection library loaded and 0 of 5 (plus 40 soak pairs) with one loaded, which
  points at a timing race. The CUPTI library we used is now in `tools/kring` (see its README for the flags); without it, expect this
  hang with the shipped launchers. Mitigation in the launchers: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`; watch the
  first requests after boot.
- **Intermittent GPTQ load check failure** (both builds): 2 of ~12 TP6 boots and 1 TP4 boot failed the loader's int4
  self-check on one rank's `shared_experts.gate_up_proj` (kernel rel error 0.07–0.15 vs tolerance 0.01), always a
  different node/layer. It did not reproduce in 12,600 isolated trials. Relaunching once has always worked.
- The TP4 KL gate passed with a 0.12-point top-1 margin (above).
