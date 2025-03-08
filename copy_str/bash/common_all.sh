PORT=8087

if [ $HOSTNAME == "toruk" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export HF_HOME=/fast_ssd/HF_DATA/
    TP_SIZE=2
    GPU_MEM=0.85
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    source /nethome/mayank/miniconda3/bin/activate
    conda activate /scratch/mayank/envs/mech_interp
    
    export PYTHONPATH=/scratch/mayank/Projects/len-gen
    export HF_HOME=/scratch/mayank/HF_DATA/
    TP_SIZE=4
    GPU_MEM=0.95
fi

export HF_TOKEN=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

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track


# For llama3.3_70B-instruct

vllm serve /scratch/common_models/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 32000 &

VLLMPID=$!

python copy_str/prompt_copy.py --ip_path datasets/500/copy_str/data.jsonl --prompt_path prompts/copy_str/zero-shot_chat_v0/ --port $PORT --config "reverse" --save_path results/copy_str/llama3.3_70B-instruct

kill $VLLMPID

sleep 5

# For llama3.1_8B-instruct

vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 32000 &

VLLMPID=$!

python copy_str/prompt_copy.py --ip_path datasets/500/copy_str/data.jsonl --prompt_path prompts/copy_str/zero-shot_chat_v0/ --port $PORT --config "reverse" --save_path results/copy_str/llama3.1_8B-instruct

kill $VLLMPID

sleep 5

# For OLMo_7B-instruct

vllm serve /scratch/common_models/OLMo-7B-0724-Instruct-hf --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 4096 --max-num-batched-tokens 32000 &

VLLMPID=$!

python copy_str/prompt_copy.py --ip_path datasets/500/copy_str/data.jsonl --prompt_path prompts/copy_str/zero-shot_chat_v0/ --port $PORT --config "reverse" --save_path results/copy_str/OLMo_7B-instruct

kill $VLLMPID
