import json
T=[json.loads(l) for l in open("out/timing.jsonl")]
A=[json.loads(l) for l in open("out/accuracy.jsonl")]
bw=json.load(open("out/membw.json"))
info=json.load(open("out/layer_info.json"))
P=json.load(open("out/profile.json"))
o=[]
o.append("# Phase 1 — live EXL3 MoE path (b12x mixed_trellis one-grid), one layer, GB10 worker node\n")
o.append("Layer 40, TP4 rank-slice geometry (H=6144, I=512/rank, 256 experts, top-8, 192 K3 + 64 K4), bf16 activations, "
         "real `Exl3MoEMethod._prepare_mixed_rank_sliced_weights` -> `_mixed_rank_sliced_runtime` -> `_apply_mixed_rank_sliced` "
         "from the live overlay exl3.py, run in image `glm53-six:e3-v2-attn-l2-v1` with the overlay bind-mounted. "
         "Env: VLLM_EXL3_PREFILL_CAPACITY=4096, max_num_batched_tokens=4096, TRELLIS_MAX_M=32 (decode plan) / block_m 32 prefill plan, GLM53_MIXED_PREFILL_CHUNK=1024.\n")
o.append("## Setup caveats (read these)\n")
o.append(f"- Only {info['n_native']}/256 experts had a fragment on node 5 for layer 40; the other {info['n_substituted']} reuse a same-K expert (whole expert incl. its own rotations; substitution map in `out/layer_info.json`). Weights are real, expert identity is not. Weight bytes per call are real either way (prepared copies are distinct memory).")
o.append("- The fragments carry PER-EXPERT hidden-side rotations (gate suh != up suh, differ across experts), so this bench runs `broadcast_suh=False, broadcast_svh=False`. If the live TP4 checkpoint is shared-H, the live kernel is a broadcast specialization; the serving head's config was not checked. Rotation bytes are ~1% of traffic so it should not matter for bandwidth.")
o.append("- The layer is a fake `SimpleNamespace` layer fed to the REAL method code (not a vLLM model), one prepared copy of the layer. L2 defeat = 256 MB `zero_()` write before every timed call (outside the timed region); GB10 L2 is far smaller than 256 MB.")
o.append("- Decode M=1..20 timed as CUDA-graph replay (one graph per M, fresh x/routes copied into static buffers each replay, 48 replays over 24 pre-generated x/route sets). Prefill M=1024/4096 eager (10 flush+call iters over up to 8 sets). Times are cuda-event time around the call only.")
o.append("- x = randn*0.5 bf16. `router` = the real layer-40 router (sigmoid + e_score_correction_bias, top8, normalize, x2.5) on that x; `distinct` = every (token,k) a distinct expert while M*8<=256 (so unique experts = 8M up to M=32), random normalized weights. Real activations probably route less uniformly than random x (fewer unique experts per M), so `router` rows are an upper-ish bound on unique experts for real traffic and `distinct` is the hard worst case.")
o.append("- GPU was otherwise idle; the NVFP4 weight download on this box was running (network/disk, not GPU). Clocks at end of bandwidth test: 2444 MHz sm.\n")
best=max(v["best_GBps"] for k,v in bw.items() if isinstance(v,dict) and "best_GBps" in v and "copy" not in k)
o.append("## Measured read bandwidth\n")
o.append("| test | best GB/s | median GB/s |\n|---|---|---|")
for k,v in bw.items():
    if isinstance(v,dict) and "best_GBps" in v: o.append(f"| {k} | {v['best_GBps']:.1f} | {v['median_GBps']:.1f} |")
o.append(f"\nRoofline used: **{best:.1f} GB/s** (best pure-read reduction; copy_ rw = 224-234 GB/s combined). Spec 273 GB/s is not achievable; ~89% of spec is what a plain reduction gets.\n")
o.append("## Accuracy: b12x mixed_trellis vs overlay per-expert `exl3_gemm` reference\n")
o.append("Reference = overlay `Exl3MoEMethod.apply` generic loop (fp16 x, `_exl3_gemm` for w1/w3/w2 per expert, silu*up, weighted index_add fp32). `bf16 floor` = rel L2 of merely rounding the fp32 reference to bf16 (the b12x output is bf16), so err at floor = agreement to output precision.\n")
o.append("| M | routing | unique experts | max abs err | ref absmax | rel L2 | bf16 floor rel L2 |\n|---|---|---|---|---|---|---|")
for r in A: o.append(f"| {r['M']} | {r['routing']} | {r['unique_experts']} | {r['max_abs_err']:.2e} | {r['ref_absmax']:.3f} | {r['rel_l2']:.2e} | {r['bf16_floor_rel_l2']:.2e} |")
o.append("\n## Timing per call\n")
o.append("weight MB = trellis + rotations of the unique experts touched (K3 slice 3.54 MB, K4 4.72 MB, +~40 KB rotations); floor = bytes / 243.2 GB/s.\n")
o.append("| M | routing | mode | us median | us min | us p90 | unique experts | weight MB | GB/s (median) | % of read BW | floor us | overhead us (med - floor) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in T:
    o.append(f"| {r['M']} | {r['routing']} | {r['mode']} | {r['us_median']:.0f} | {r['us_min']:.0f} | {r['us_p90']:.0f} | {r['unique_experts']:.1f} | {r['weight_bytes']/1e6:.1f} | {r['GBps_median']:.0f} | {r['pct_roofline']:.0f}% | {r['floor_us']:.0f} | {r['us_median']-r['floor_us']:.0f} |")
# linear fits
import statistics
def fit(kind):
    pts=[(r['weight_bytes']/1e6,r['us_median']) for r in T if r['mode']=='graph' and r['routing']==kind]
    n=len(pts); mx=sum(p[0] for p in pts)/n; my=sum(p[1] for p in pts)/n
    b=sum((p[0]-mx)*(p[1]-my) for p in pts)/sum((p[0]-mx)**2 for p in pts); a=my-b*mx
    return a,b
o.append("\n### Linear fit of decode graph time vs bytes (M=1..20)\n")
for kind in ("router","distinct"):
    a,b=fit(kind); o.append(f"- {kind}: t = {a:.0f} us + {b:.2f} us/MB  (marginal {1e3/b:.0f} GB/s = {100*1e3/b/243.2:.0f}% of read BW)")
o.append("\nPrefill compute check: M=1024 = 8192 (token,expert) rows x 3 x 2 x 512 x 6144 = 155 GFLOP in 10.8 ms = 14 TFLOP/s; M=4096 = 620 GFLOP in 31.7 ms = 20 TFLOP/s. Prefill is compute/dequant-bound, not weight-read-bound, so % of read BW is not the right yardstick there.")
o.append("\n## Kernel breakdown (torch profiler, 10 flush+call iterations)\n")
o.append("Per MoE call, kernels launched: route pack (2 tiny kernels), one `W4A16MixedTrellisKernel` grid (FC1+act+FC2), `W4A16TopKSumKernel`, one bf16 dtype-copy. No other torch glue (rotations run inside the one grid).\n")
o.append("| case | wall span first->last kernel | route pack (prefix+sort) | main mixed_trellis grid | topk_sum | bf16 copy |\n|---|---|---|---|---|---|")
for key,v in P.items():
    d={}
    for row in v["summary"]:
        k=row["kernel"]
        if "MOE_CALL" in k: continue
        n=("main" if "MixedTrellisKernel" in k else "topk" if "TopKSum" in k else "pack" if "pack_topk" in k else "copy" if "copy" in k else "other")
        d[n]=d.get(n,0)+row["us_per_moe"]
    o.append(f"| {key} | {v['span_us_first_to_last_kernel']:.0f} us | {d.get('pack',0):.1f} us | {d.get('main',0):.0f} us | {d.get('topk',0):.1f} us | {d.get('copy',0):.1f} us |")
o.append("\n(Profiler runs a bit slower than the event timing above; eager first kernel starts ~250 us after call entry = CPU launch latency, hidden in graph mode.)\n")

# ---------------- Phase 2 ----------------
MM=json.load(open("out/mm_peak.json"))
E3=[json.loads(l) for l in open("out/e3.jsonl")]
BS={b:[json.loads(l) for l in open(f"out/blocksweep_{b}.jsonl")] for b in (32,64,128)}
o.append("\n# Phase 2 — prefill: tensor-core ceiling, block_m sweep, live E3 path, ncu\n")
o.append("Same fake layer-40 harness and L2-flush method as Phase 1 (`bench_layer.py`). Prefill is eager. E3 = Mia's live `TP4_E3_PREFILL` path (gather + gateup + down kernels), run from its package on one worker node (cubin rebuilt with `nvcc -arch=sm_121a`, so not byte-identical to the manifest cubin); measured, not copied.\n")
o.append("## A. Dense tensor-core peak on one worker node (torch `mm`, best of N)\n")
o.append("| op | M | N | K | dtype | best TFLOP/s | median TFLOP/s |\n|---|---|---|---|---|---|---|")
for r in MM: o.append(f"| {r['op']} | {r['M']} | {r['N']} | {r['K']} | {r['dtype']} | {r['best_TFLOPs']:.1f} | {r['median_TFLOPs']:.1f} |")
o.append("\nfp16/bf16 dense peak ~100 TFLOP/s. The two per-expert GEMM shapes (gate/up: N=1024,K=6144 ; down: N=6144,K=512) reach ~93 / ~64 TFLOP/s at M=4096 and 42-83 at M=128-512. e4m3 `_scaled_mm` is 213 TFLOP/s (2x).\n")
o.append("## B. b12x prefill `block_m` sweep (one-grid mixed_trellis, router routing)\n")
o.append("| M | block_m | us median | main grid us | route+topk us | rel L2 |\n|---|---|---|---|---|---|")
for b,rows in BS.items():
    for r in rows:
        if "error" in r: o.append(f"| {r['M']} | {b} | FAILED: `{r['error'][:90]}` | | | |")
        else: o.append(f"| {r['M']} | {b} | {r['us_median']:.0f} | {r['main_grid_us']:.0f} | {r['route_topk_us_total']:.0f} | {r['rel_l2']:.2e} |")
o.append("\nblock_m=32 is fastest at every M; 64 is 5-20% slower; 128 is rejected at planning by the FC2 subtile check. Accuracy identical for 32/64. Main grid is ~28.5 of 31.7 ms at M=4096.\n")
o.append("## C. E3 (live prefill path) vs b12x prefill\n")
b32={r['M']:r for r in BS[32]}
o.append("Per-kernel us are torch-profiler GPU times per call; `fill/memset` row in `out/e3.jsonl` also contains the 256 MB L2-flush fill (~1.3-1.9 ms) and is excluded here.\n")
o.append("| M | E3 wall us | b12x wall us (block_m 32) | E3 speedup | gather us | gateup us | down us | glue+zero+cast us (rest) | rel L2 vs ref | rel L2 E3 vs b12x |\n|---|---|---|---|---|---|---|---|---|---|")
for r in E3:
    k=r["kernels_us_per_call"]; g,gu,d=k["gather"],k["gateup"],k["down"]
    rest=sum(v for kk,v in k.items() if kk.startswith("glue"))
    bw=b32.get(r["M"]); bws=f"{bw['us_median']:.0f}" if bw else "n/a"; sp=f"{bw['us_median']/r['us_median']:.2f}x" if bw else "n/a"
    o.append(f"| {r['M']} | {r['us_median']:.0f} | {bws} | {sp} | {g:.0f} | {gu:.0f} | {d:.0f} | {rest:.0f} | {r['rel_l2']:.2e} | {r['rel_l2_vs_b12x']:.2e} |")
o.append("\nAt M=4096 E3 is 22.4 ms vs 31.7 ms (1.42x, 29% less time). Its 3 big kernels are 4.4 + 7.8 + 8.5 = 20.8 ms; glue (routing sort/scan, zero_, bf16 cast) is ~0.9 ms. Compute achieved ~30 TFLOP/s (620 GFLOP / 20.8 ms) vs ~64-93 TFLOP/s dense fp16 on the same shapes.\n")
o.append("## D. ncu on E3 at M=4096 (`out/ncu_e3_4096.csv`, `out/ncu_e3_full_4096.csv`; view: `python3 ncu_show.py <csv>`)\n")
o.append("| kernel | duration | grid | regs/thread | occupancy | issue slots busy | SM busy | shared bank conflicts | cycles / issued instr |\n|---|---|---|---|---|---|---|---|---|")
o.append("| gateup | 7.77 ms | 2048 | 128 | 33% (2 blocks/SM; limit = registers and 48 KB smem) | 45% | 64% | 3.5 M | 9.5 |")
o.append("| down | 8.45 ms | 12288 | 128 | 33% | 24% | 29% | 20.8 M | 17.9 |")
o.append("| gather | 4.5 ms | 49152 | 39 | n/a | n/a | n/a | n/a | n/a |")
o.append("\ngateup: ALU pipe 24%, fmaheavy 22.5%, LSU 40%, Mem Pipes Busy 39.8%. down: Mem Pipes Busy 21.3%. gather writes 2 x 400 MB fp16 copies (LSU 36.6%).\n")
o.append("Not obtainable on GB10 with this ncu: `sm__pipe_tensor_op_hmma_*` and every `dram__*` metric come back `n/a`, and the full set has no tensor-pipe row, so tensor-pipe and DRAM utilisation of E3 are unmeasured. Reading from what is measured: both big kernels are latency/issue-bound (33% occupancy, register + smem limited, 9.5-18 cycles per issued instruction), neither DRAM-bound nor evidently tensor-bound. `down` is worse than `gateup` (24% vs 45% issue slots, 6x the bank conflicts). M=5 b12x ncu was not run (optional).\n")

open("RESULTS.md","w").write("\n".join(o)+"\n")
