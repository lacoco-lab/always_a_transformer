#!/bin/bash

# Conda/venv commands
source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

# HF model name, appropriate tensor parallel size, ideally check all the parameters
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /scratch/common_models/Llama-3.1-70B --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port 8080 &

# We want to shut down the VLLM server after the experiment is done, so we need its PID
VLLMPID=$!

BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_70B/distance"

# Iterate over all files in the input directory
for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai
    fi
  done
done

# Shut down the VLLM server
kill $VLLMPID

