# Conda/venv commands
#source /scratch/yanav/anaconda3/bin/activate
#conda activate len-gen

## HF model name, appropriate tensor parallel size, ideally check all the parameters
#CUDA_VISIBLE_DEVICES=0,1 vllm serve allenai/OLMo-7B-0724-hf --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port 8080 & 
#
## We want to shut down the VLLM server after the experiment is done, so we need its PID
#VLLMPID=$!

# --------------------------------------------------------------------------------------------
# DISTANCE COMPLETION OLMo-7B
# --------------------------------------------------------------------------------------------

#BASE_INPUT_DIR="datasets/flipflop/distance"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/distance"
#PROMPT_PATH="prompts/flipflop_zero-shot_completion_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#
#BASE_INPUT_DIR="datasets/flipflop/distance-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/distance-worded"
#PROMPT_PATH="prompts/flipflop_zero-shot_completion_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#
#BASE_INPUT_DIR="datasets/flipflop/distance"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/distance-qa"
#PROMPT_PATH="prompts/flipflop_qa_zero-shot_completion_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#
#BASE_INPUT_DIR="datasets/flipflop/distance-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/distance-qa-worded"
#PROMPT_PATH="prompts/flipflop_qa_zero-shot_completion_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/distance"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/distance-masked"
#PROMPT_PATH="prompts/flipflop_mask_zero-shot_completion_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/distance-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/distance-masked-worded"
#PROMPT_PATH="prompts/flipflop_mask_zero-shot_completion_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
## --------------------------------------------------------------------------------------------
## SPARSE COMPLETION OLMo-7B
## --------------------------------------------------------------------------------------------
#
#BASE_INPUT_DIR="datasets/flipflop/sparse"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/sparse"
#PROMPT_PATH="prompts/flipflop_zero-shot_completion_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/sparse-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/sparse-worded"
#PROMPT_PATH="prompts/flipflop_zero-shot_completion_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/sparse"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/sparse-qa"
#PROMPT_PATH="prompts/flipflop_qa_zero-shot_completion_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/sparse-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/sparse-qa-worded"
#PROMPT_PATH="prompts/flipflop_qa_zero-shot_completion_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done

#BASE_INPUT_DIR="datasets/flipflop/sparse"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/sparse-masked"
#PROMPT_PATH="prompts/flipflop_mask_zero-shot_completion_v0"
#
#for SUBFOLDER in s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/sparse-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B/sparse-masked-worded"
#PROMPT_PATH="prompts/flipflop_mask_zero-shot_completion_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B"
#
#      python flipflop/complete_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
## Shut down the VLLM server
#kill $VLLMPID
#
#sleep 10s

# --------------------------------------------------------------------------------------------
# DISTANCE INSTRUCT OLMo-7B
# --------------------------------------------------------------------------------------------

vllm serve allenai/OLMo-7B-0724-Instruct-hf --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port 8080 &

# We want to shut down the VLLM server after the experiment is done, so we need its PID
VLLMPID=$!

#BASE_INPUT_DIR="datasets/flipflop/distance"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/distance"
#PROMPT_PATH="prompts/flipflop_zero-shot_chat_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/distance-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/distance-worded"
#PROMPT_PATH="prompts/flipflop_zero-shot_chat_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/distance"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/distance-qa"
#PROMPT_PATH="prompts/flipflop_qa_zero-shot_chat_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/distance-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/distance-qa-worded"
#PROMPT_PATH="prompts/flipflop_qa_zero-shot_chat_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/distance"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/distance-masked"
#PROMPT_PATH="prompts/flipflop_mask_zero-shot_chat_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done

#BASE_INPUT_DIR="datasets/flipflop/distance-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/distance-masked-worded"
#PROMPT_PATH="prompts/flipflop_mask_zero-shot_chat_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
## --------------------------------------------------------------------------------------------
## SPARSE INSTRUCT OLMo-7B
## --------------------------------------------------------------------------------------------
#
#BASE_INPUT_DIR="datasets/flipflop/sparse"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/sparse"
#PROMPT_PATH="prompts/flipflop_zero-shot_chat_v0"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done
#
#BASE_INPUT_DIR="datasets/flipflop/sparse-worded"
#BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/sparse-worded"
#PROMPT_PATH="prompts/flipflop_zero-shot_chat_worded"
#
#for SUBFOLDER in s1 s2 s3 s4 s5; do
#  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
#  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  for INPUT_FILE in "$INPUT_DIR"/*; do
#    if [[ -f $INPUT_FILE ]]; then
#      BASENAME=$(basename "$INPUT_FILE" .txt)
#
#      echo "Processing file: $INPUT_FILE"
#      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"
#
#      python flipflop/prompt_flipflop.py \
#        --ip_path "$INPUT_FILE" \
#        --save_path "$OUTPUT_DIR" \
#        --engine openai \
#        --prompt_path "$PROMPT_PATH"
#    fi
#  done
#done

BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/sparse-qa"
PROMPT_PATH="prompts/flipflop_qa_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH"
    fi
  done
done

BASE_INPUT_DIR="datasets/flipflop/sparse-worded"
BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/sparse-qa-worded"
PROMPT_PATH="prompts/flipflop_qa_zero-shot_chat_worded"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH"
    fi
  done
done

BASE_INPUT_DIR="datasets/flipflop/sparse"
BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/sparse-masked"
PROMPT_PATH="prompts/flipflop_mask_zero-shot_chat_v0"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH"
    fi
  done
done

BASE_INPUT_DIR="datasets/flipflop/sparse-worded"
BASE_OUTPUT_DIR="results/flipflop/OLMo_7B-instruct/sparse-masked-worded"
PROMPT_PATH="prompts/flipflop_mask_zero-shot_chat_worded"

for SUBFOLDER in s1 s2 s3 s4 s5; do
  INPUT_DIR="$BASE_INPUT_DIR/$SUBFOLDER"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/$SUBFOLDER"

  mkdir -p "$OUTPUT_DIR"

  for INPUT_FILE in "$INPUT_DIR"/*; do
    if [[ -f $INPUT_FILE ]]; then
      BASENAME=$(basename "$INPUT_FILE" .txt)

      echo "Processing file: $INPUT_FILE"
      echo "Running Prompt: $PROMPT_PATH with model: OLMo-7B-instruct"

      python flipflop/prompt_flipflop.py \
        --ip_path "$INPUT_FILE" \
        --save_path "$OUTPUT_DIR" \
        --engine openai \
        --prompt_path "$PROMPT_PATH"
    fi
  done
done

# Shut down the VLLM server
kill $VLLMPID
