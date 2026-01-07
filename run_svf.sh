#!/bin/bash

# Parse arguments
DEBUG=0
PROFILE=0
SPEC=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            DEBUG=1
            shift
            ;;
        -p|--profile)
            PROFILE=1
            shift
            ;;
        --spec)
            SPEC="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Debug mode: enable CUDA coredump
if [ "$DEBUG" -eq 1 ]; then
    export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
    export CUDA_COREDUMP_SHOW_PROGRESS=1
    export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
    export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
fi

# Profile mode: nsys profiling
PROFILE_CMD=""
if [ "$PROFILE" -eq 1 ]; then
    PROFILE_CMD="nsys profile \
        -o /home/yongye/vllm/traces/svf_%h_%p \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end repeat"
fi

# Build engine args
ENGINE_ARGS=(
    --port 8001
    -dp 4 -ep
    --kv-cache-dtype fp8
    --block-size 256
    --max_model_len auto
    --reasoning-parser deepseek_v3
    --no-enable-flashinfer-autotune
    -O0
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)

# Profile: add profiler config
if [ "$PROFILE" -eq 1 ]; then
    ENGINE_ARGS+=(--profiler-config '{"profiler":"cuda","delay_iterations":30,"max_iterations":100}')
fi

# Speculative decoding config
if [ "$SPEC" -eq 1 ] || [ "$SPEC" -eq 3 ]; then
    ENGINE_ARGS+=(
        --speculative-config.method deepseek_mtp
        --speculative-config.num_speculative_tokens "$SPEC"
    )
fi

$PROFILE_CMD vllm serve /mnt/lustre/svf-2026-04/ckpt20260409/DeepSeek-V4-HF-FP4/ \
    "${ENGINE_ARGS[@]}"


# lm_eval --model local-completions \
#   --model_args "model=/mnt/lustre/svf-2026-04/ckpt20260409/DeepSeek-V4-HF-FP4/,base_url=http://0.0.0.0:8001/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=2048,timeout=5000,max_length=32768" \
#   --tasks gsm8k \
#   --num_fewshot 5 \
#   --log_samples \
#   --output_path "gsm8k_results"
