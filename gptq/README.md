# D8: GPTQ int4 g32 codes for the dense decode path

The D8 overlay (`overlays/dense-fp8-gptq/d2_fp8_gptq.py`) runs 312 non-expert linears per rank (attention
`q_a` part of `fused_qkv_a`, `o_proj`, shared-expert and dense-layer `gate_up`/`down`) as int4 (symmetric,
group 32, per-group MSE clip) through Marlin whenever a batch has fewer than 256 rows, i.e. every decode and
MTP-verify step. Prefill keeps the FP8/W8A8 path. Cost: ~+1.9 GB per rank (TP6).

The codes are **rank-specific** (they are solved for each rank's slice of each weight) and are not shipped.
You regenerate them with the scripts here. We did it off-cluster on one large-RAM x86 host (CPU only).

## Steps

1. **Calibration token ids.** `tp6/make_calib.py` turns `corpus.txt` (documents separated by `\n\x00DOC\x00\n`)
   into `calib_ids.jsonl`: 1,024-token chunks with the chat-template prefix ids. It needs `src/tokenizer.json`
   from the checkpoint. **Our corpus is private and not included** (~108k tokens of our own internal text).
   Bring your own; the solved codes, and so the gate numbers, depend on it.
2. **Held-out sequences.** `seqs.jsonl` = `eval/kl-seqs.jsonl` (16 greedy continuations, also the KL gate set).
3. **Capture boot.** Boot the stack with `d2_fp8_capture.py` bound in place of `d2_fp8.py`, plus
   `GLM_D8_W4=0 GLM_D8_CAPTURE_DIR=/d8cap GLM_D8_CAPTURE_FULL=1 GLM_D8_CAPTURE_TP=<6|4>`, a writable host dir
   bound at `/d8cap` on every node, `--enforce-eager`, `cudagraph_mode NONE`, `--no-enable-prefix-caching`,
   `--max-model-len 32768` and a small KV budget (we used 6–30 GB). Then run `tp6/run_capture.py`
   (or `tp4/run_capture_tp4.py`) with `HEAD_SSH`, `CAPTURE_DIR` and `RANK_HOSTS` set. It sends the calibration
   prefills, then the held-out ones, then triggers the per-rank dump of input Hessians and weight fingerprints.
   TP6: ~5 min, 6.5–7.8 GB of dumps per rank. `make_capture_overlay.py` / `make_gptq_overlay.py` regenerate the
   two overlays from `d2_fp8.py`.
4. **Dense BF16 sources.** `tp6/extract_dense.py model-layer-LLL.safetensors dense/layer-LLL.safetensors` for all
   78 layers (from the rank-0 TP6 export or the TP4 checkpoint; the dense tensors are unsharded there).
5. **Solve.** TP6: `tp6/solve_gptq.py --dense dense --cap cap --out out32 --group 32` (1,482 modules, ~47 min on our
   host), then `tp6/fix_outliers.py out32 dense cap 32` (re-solves the few modules whose dequantized weight error
   exceeds the loader's 0.2 limit with more damping). TP4: `tp4/solve_gptq_tp4.py --dense dense --cap cap4
   --cap-rep cap6 --rep-tp 6 --out out32 --group 32` (reuses the TP6 solver, pinned by `tp4/tp6_solver.sha256`,
   with the TP4 slicing and inline outlier re-solve).
6. **Stage.** Copy `out32/r<rank>/` (78 `layer-LLL.safetensors`) to each node and point that node's `GPTQ_DIR` at it.
   The loader checks a fingerprint of the live weights against each file and refuses a mismatched rank.
7. **Gate.** Run the KL gate (`eval/README.md`) before serving.
