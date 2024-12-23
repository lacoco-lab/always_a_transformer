# Conda/venv commands
source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

# Replace LD_LIBRARY_PATH with the path to the nvjitlink library in the conda environment (not sure if this is needed/not)
export LD_LIBRARY_PATH=/scratch/yanav/anaconda3/lib/python3.12/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH

# HF model name, appropriate tensor parallel size, ideally check all the parameters
CUDA_VISIBLE_DEVICES=0 vllm serve /scratch/common_models/OLMo-7B-0724-Instruct-hf --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port 8080 &

# We want to shut down the VLLM server after the experiment is done, so we need its PID
VLLMPID=$!

INPUT_DIR="datasets/flipflop/distance/s3"
OUTPUT_DIR="results/flipflop/OLMo_7B/distance/s3"

# Iterate over all files in the input directory
for INPUT_FILE in "$INPUT_DIR"/*; do
  if [[ -f $INPUT_FILE ]]; then
    # Extract filename without extension for output naming
    BASENAME=$(basename "$INPUT_FILE" .txt)

    echo "Processing file: $INPUT_FILE"


    # Run the experiment, possible to run multiple experiments in sequence
    python flipflop/prompt_flipflop.py \
      --ip_path "$INPUT_FILE" \
      --save_path "$OUTPUT_DIR" \
      --engine openai
  fi
done

# Shut down the VLLM server
kill $VLLMPID

