#!/bin/bash

# Set the base command
BASE_CMD="HF_HOME=/mnt/local/yongye python benchmark_moe.py --tune --trust_remote_code"

# Define the tensor parallel sizes and models
TP_SIZES=(1 4 8)
MODELS=(
  # "Qwen/Qwen3-30B-A3B"
  # "deepseek-ai/DeepSeek-V2-Lite"
  # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
)

# Loop over each combination
for model in "${MODELS[@]}"; do
  for tp in "${TP_SIZES[@]}"; do
    echo "Running with model=$model, tp-size=$tp"
    eval $BASE_CMD --tp-size=$tp --model="$model"
  done
done
