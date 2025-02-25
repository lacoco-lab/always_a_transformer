#!/bin/bash

# Conda/venv commands
#source /scratch/yanav/anaconda3/bin/activate
#conda activate len-gen
source /nethome/mayank/miniconda3/bin/activate
conda activate /scratch/mayank/envs/mech_interp

export HF_HOME=/scratch/mayank/HF_DATA/
export HF_TOKEN=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
export PYTHONPATH=/scratch/mayank/Projects/len-gen

new_devices=""
IFS=',' read -ra my_array <<< "$CUDA_VISIBLE_DEVICES"
for id in ${my_array[@]};
do
    new_devices=${new_devices}`nvidia-smi -L | grep $id | sed -E "s/^GPU ([0-9]+):.*$/\1/"`,
done
export CUDA_VISIBLE_DEVICES=${new_devices%?}
echo "now: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# HF model name, appropriate tensor parallel size, ideally check all the parameters
PORT=8087
vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port "$PORT" &

# We want to shut down the VLLM server after the experiment is done, so we need its PID
VLLMPID=$!

# --------------------------------------------------------------------------------------------
# DISTANCE COMPLETION Llama-3.1-8B
# --------------------------------------------------------------------------------------------

BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B/distance"
PROMPT_PATH="prompts/flipflop_zero-shot_completion_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B/distance-qa"
PROMPT_PATH="prompts/flipflop_qa_zero-shot_completion_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B/distance-masked"
PROMPT_PATH="prompts/flipflop_mask_zero-shot_completion_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


# --------------------------------------------------------------------------------------------
# SPARSE COMPLETION Llama-3.1-8B
# --------------------------------------------------------------------------------------------

BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B/sparse"
PROMPT_PATH="prompts/flipflop_zero-shot_completion_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B/sparse-qa"
PROMPT_PATH="prompts/flipflop_qa_zero-shot_completion_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B/sparse-masked"
PROMPT_PATH="prompts/flipflop_mask_zero-shot_completion_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B"

      python flipflop/complete_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


# Shut down the VLLM server
kill $VLLMPID

sleep 10s

# --------------------------------------------------------------------------------------------
# DISTANCE INSTRUCT Llama-3.1-8B-instruct
# --------------------------------------------------------------------------------------------

# HF model name, appropriate tensor parallel size, ideally check all the parameters
PORT=8087
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port "$PORT" &

# We want to shut down the VLLM server after the experiment is done, so we need its PID
VLLMPID=$!

BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B-instruct/distance"
PROMPT_PATH="prompts/flipflop_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B-instruct/distance-qa"
PROMPT_PATH="prompts/flipflop_qa_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/distance"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B-instruct/distance-masked"
PROMPT_PATH="prompts/flipflop_mask_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done

# --------------------------------------------------------------------------------------------
# SPARSE INSTRUCT Llama-3.1-8B-instruct
# --------------------------------------------------------------------------------------------

BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B-instruct/sparse"
PROMPT_PATH="prompts/flipflop_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B-instruct/sparse-qa"
PROMPT_PATH="prompts/flipflop_qa_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done


BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/llama3.1_8B-instruct/sparse-masked"
PROMPT_PATH="prompts/flipflop_mask_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: Llama-3.1-8B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH" \
        --port "$PORT"
    fi
  done
done

# Shut down the VLLM server
kill $VLLMPID
