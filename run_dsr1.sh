#!/usr/bin/bash

# Usage:
#   Master: ./run_prefill.sh --dp 8
#   Worker: ./run_prefill.sh --dp 8 --dp-start-rank 4 --master-addr 172.27.54.178
#
# With hybrid LB, BOTH master and worker run API servers on port 8000.
# The router must include both endpoints:
#   --prefill http://<master>:8000 --prefill http://<worker>:8000

IP=$(hostname -I | awk '{print $1}')

# Defaults
DP=4
DP_LOCAL=4
DPSR=0
MASTER_ADDR=$IP
PROFILE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dp)
            DP="$2"
            shift 2
            ;;
        --dp-local)
            DP_LOCAL="$2"
            shift 2
            ;;
        --dp-start-rank)
            DPSR="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --profile|-p)
            PROFILE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "launching prefill node in ${IP}"
echo "DP=$DP, DP_LOCAL=$DP_LOCAL, DP_START_RANK=$DPSR, MASTER_ADDR=$MASTER_ADDR"

# Backend params
export VLLM_ENABLE_MOE_DP_CHUNK=0
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1

# Model Related
export HF_HOME="${HF_HOME:=/mnt/lustre/hf-models}"

# ---- Library Fix ----
# CRITICAL: Fix corrupted HPC-X UCX library (libuct_ib_mlx5.so.0.0.0)
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/hpcx' | paste -sd ':')
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/opt/hpcx' | paste -sd ':')

# ---- UCX Configuration for MNNVL ----
export UCX_MEMTYPE_CACHE=n
export UCX_MEMTYPE_REG_WHOLE=n
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export UCX_CUDA_IPC_ENABLE_MNNVL=y

# UCX timeouts for stability
export UCX_RC_TIMEOUT=5s
export UCX_RC_RETRY_COUNT=14
export UCX_RC_RNR_TIMEOUT=10ms
export UCX_RC_RNR_RETRY_COUNT=14

# ---- GLOO ----
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:=enp192s2}"

# ---- NCCL Configuration for GB200 NVL72 ----
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:=enp192s2}"
export NCCL_IB_HCA="mlx5_1,mlx5_2,mlx5_3,mlx5_4"
export NCCL_NET_GDR_LEVEL=5
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN

# ---- NIXL Side Channel ----
export VLLM_NIXL_SIDE_CHANNEL_HOST=${IP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600

# ---- PDL ----
export VLLM_ENABLE_PDL=1

# Engine arguments
MODEL=nvidia/Kimi-K2.5-NVFP4
# -O3 \
# COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
# --compilation-config "$COMPILATION_CONFIG" \
# --load-format fastsafetensors \
ENGINE_ARGS="
    -dp $DP \
    -ep \
    --enforce-eager \
    --trust-remote-code \
    --language-model-only \
    --attention_config.disable_flashinfer_prefill false \
    --attention_config.use_prefill_query_quantization true \
    --attention_config.use_trtllm_ragged_deepseek_prefill true \
    --gpu-memory-utilization 0.85 \
    --data-parallel-size-local $DP_LOCAL \
    --data-parallel-address $MASTER_ADDR \
    --data-parallel-hybrid-lb \
    --max-model-len 9216 \
    --max-num-batched-tokens 16384 \
    --kv-transfer-config {\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"fail\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"enforce_handshake_compat\":false}} \
    --disable-uvicorn-access-log \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching \
"

PROFILE_CMD=""
if [ "$PROFILE" -eq 1 ]; then
    PROFILE_CMD="nsys profile \
        -o /home/yongye/vllm/traces/dsr1_%h_%p \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end repeat"
    ENGINE_ARGS+=" --profiler-config {\"profiler\":\"cuda\",\"delay_iterations\":30,\"max_iterations\":100}"
fi

# Needed for MNNVL
ENGINE_ARGS+=" --enable-sleep-mode"
# Worker nodes need --data-parallel-start-rank
if [ "$DPSR" -gt 0 ]; then
    ENGINE_ARGS+=" --data-parallel-start-rank $DPSR"
fi
# Use LOG_DIR and hostname to avoid collision when multiple instances run
LOG_FILE="${LOG_DIR:-.}/prefill-worker-${DPSR}-$(hostname -s).log"
$PROFILE_CMD vllm serve $MODEL $ENGINE_ARGS 2>&1 | tee "$LOG_FILE"