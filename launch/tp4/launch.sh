#!/usr/bin/env bash
# GLM 5.3 full, EXL3 TR3 3.25 bpw (native TP4 export), TP4 on four DGX Spark (GB10) nodes: the kept TP4 stack.
#   control (DCP2 ag_rs, MTP k=4 FULL 5/10/15/20, E3 grouped-fragment prefill, prepared-weight cache, fastload)
#   + D2d FP8/W8A8 dense prefill + dual-HCA NCCL + D3 RoCE one-shot AR/AG + D8 GPTQ int4 g32 dense decode + Marlin atomic-add.
# Run once per node with that node's RANK: workers 1..3 first, head (rank 0) last.
# Reconstructed from the recorded container config of the measured run; see README.md "Caveats".
# Settings come from the environment (see launch.env.example). DRY_RUN=1 (default) only prints the command.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=${REPO:-$(cd "$here/../.." && pwd)}
need() { for v; do [ -n "${!v:-}" ] || { echo "set $v (see launch.env.example)" >&2; exit 2; }; done; }
need IMAGE MODEL_DIR GPTQ_DIR CACHE_DIR RANK HEAD_ADDR HOST_ADDR NET_IFACE IB_HCA
[[ "$RANK" =~ ^[0-3]$ ]] || { echo 'RANK must be 0..3' >&2; exit 2; }
for p in "$MODEL_DIR" "$GPTQ_DIR"; do [ -e "$p" ] || { echo "missing: $p" >&2; exit 2; }; done
n=$(ls "$GPTQ_DIR"/layer-*.safetensors 2>/dev/null | wc -l); [ "$n" = 78 ] || { echo "GPTQ_DIR has $n layer files, want 78" >&2; exit 2; }

V=/usr/local/lib/python3.12/dist-packages/vllm
O=$REPO/overlays
binds=(
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/root/.cache" -v "$GPTQ_DIR:/d8gptq-w:ro"
  -v "$O/vllm-common/kernel_warmup_graphs.py:$V/model_executor/warmup/kernel_warmup.py:ro"
  # TP4 variant of the route-pack warmup (DCP draft-slot aware)
  -v "$O/vllm-tp4/tr3_boot_warmup.py:$V/model_executor/warmup/tr3_boot_warmup.py:ro"
  # prepared-weight cache loader + fastload (entrypoint, O_DIRECT restore, weight filter)
  -v "$O/tp4-load-accel:/opt/tr3-load-accel:ro"
  -v "$O/tp4-load-accel/ep_weight_filter.py:$V/model_executor/model_loader/ep_weight_filter.py:ro"
  # E3 grouped-fragment prefill runtime (AGPL-3.0, see third_party/e3)
  -v "$REPO/third_party/e3:/opt/tp4-e3/e3:ro" -v "$REPO/third_party/e3/sitecustomize.py:/opt/tp4-e3/sitecustomize.py:ro"
  # DCP2 on the SM120 sparse-MLA backend + DCP-aware MTP draft slots
  -v "$O/vllm-tp4/sm120_dcp.py:$V/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:ro"
  -v "$O/vllm-tp4/mla_attn_dcp.py:$V/model_executor/layers/attention/mla_attention.py:ro"
  -v "$O/vllm-tp4/spec_decode_utils.py:$V/v1/spec_decode/utils.py:ro"
  -v "$O/vllm-tp4/llm_base_proposer.py:$V/v1/spec_decode/llm_base_proposer.py:ro"
  # D2d + D8: EXL3 hook, MTP eh_proj fix, FP8/W8A8 dense + GPTQ int4 decode
  -v "$O/vllm-tp4/exl3.py:$V/model_executor/layers/quantization/exl3.py:ro"
  -v "$O/vllm-tp4/deepseek_mtp.py:$V/model_executor/models/deepseek_mtp.py:ro"
  -v "$O/dense-fp8-gptq/d2_fp8_gptq.py:/opt/tr3/d2_fp8.py:ro"
  # D3: RoCE one-shot all-reduce / all-gather
  -v "$O/d3-roce/vllm/cuda_communicator.py:$V/distributed/device_communicators/cuda_communicator.py:ro"
  -v "$O/d3-roce/vllm/b12x_roce_all_reduce.py:$V/distributed/device_communicators/b12x_roce_all_reduce.py:ro"
  -v "$O/d3-roce/vllm/parallel_state.py:$V/distributed/parallel_state.py:ro"
  -v "$O/d3-roce/vllm/envs.py:$V/envs.py:ro"
  -v "$O/d3-roce/vllm/gpu_worker.py:$V/v1/worker/gpu_worker.py:ro"
  -v "$O/d3-roce/b12x_comm_roce:/opt/tr3/b12x/comm/roce:ro"
)
envs=(
  -e VLLM_HOST_IP="$HOST_ADDR" -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
  -e PYTHONPATH=/opt/tp4-e3:/opt/tr3-load-accel:/opt/tr3 -e PYTHONPYCACHEPREFIX=/root/.cache/pycache
  -e NCCL_SOCKET_IFNAME="$NET_IFACE" -e GLOO_SOCKET_IFNAME="$NET_IFACE"
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_MERGE_NICS=1
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG=WARN -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_CROSS_NIC=1
  -e VLLM_EXL3_PREFILL_CAPACITY=4096 -e GLM53_MIXED_PREFILL_CHUNK=1024
  -e TP4_E3_PREFILL=1 -e TP4_DCP_DRAFT_SLOTS=1 -e GLM6_E3_AUDIT=0 -e EXL3_SM120_LSE_BASE_E=0
  -e TR3_RANK="$RANK" -e TR3_LOAD_ACCEL=1 -e TR3_FAST=1 -e TR3_SKIP_FOREIGN_RANKS=1
  -e TR3_BUILD_PREPARED=1 -e TR3_PREPARED_CACHE=/root/.cache/tr3-prepared -e TR3_PREP_WORKERS=2
  -e B12X_COMPILE_CACHE_DIR=/root/.cache/b12x/compile -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
  -e TRITON_CACHE_DIR=/root/.cache/triton -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
  -e GLM_D2_FP8=1 -e GLM_D2_FP8_BIG=w8a8 -e GLM_D2_FP8_DEQUANT_M=256
  -e GLM_D8_W4=1 -e GLM_D8_W4_GROUP=32 -e GLM_D8_GPTQ_DIR=/d8gptq-w -e VLLM_MARLIN_USE_ATOMIC_ADD="${ATOMIC:-1}"
  -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=256KB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=2MB
)
[ -n "${FULLVERIFY:-}" ] && envs+=(-e FULLVERIFY="$FULLVERIFY")  # force the full prepared-cache sha verify (first boot after a copy)
args=(/opt/tr3-load-accel/entrypoint.py serve /model --served-model-name glm-5.3 --host 0.0.0.0 --port 8000
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$RANK" --master-addr "$HEAD_ADDR" --master-port 29561
  --distributed-executor-backend mp --disable-custom-all-reduce
  --quantization exl3 --dtype bfloat16 --kv-cache-dtype fp8_ds_mla --attention-backend FLASHINFER_MLA_SPARSE_SM120
  --max-model-len 360000 --max-num-seqs 4 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 20000000000
  --enable-prefix-caching --no-enable-flashinfer-autotune
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
  --decode-context-parallel-size 2
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":4}'
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[5,10,15,20]}'
  --jit-monitor-verbose)
[ "$RANK" = 0 ] || args+=(--headless)
cmd=(docker run -d --name "${NAME:-glm53-tp4}" --restart no --gpus all --network host --ipc host
  --cap-add IPC_LOCK --security-opt label=disable --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576
  --device /dev/infiniband:/dev/infiniband "${binds[@]}" "${envs[@]}" --entrypoint python3 "$IMAGE" "${args[@]}")
printf '%q ' "${cmd[@]}"; echo
[ "${DRY_RUN:-1}" = 1 ] && { echo '(DRY_RUN=1: printed only)' >&2; exit 0; }
"${cmd[@]}"
