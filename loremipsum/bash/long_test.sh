PORT=8089

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

python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data_3000_tokens.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "exact"
python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data_3000_tokens.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "verbatim"

python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data_4000_tokens.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "exact"
python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data_4000_tokens.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "verbatim"

python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data_5000_tokens.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "exact"
python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data_5000_tokens.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "verbatim"

kill $VLLMPID

sleep 5