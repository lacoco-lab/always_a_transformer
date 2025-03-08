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

# For llama3.1_70

vllm serve /scratch/common_models/Llama-3.1-70B/ --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 256000 &

VLLMPID=$!

python loremipsum/complete_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_completion_v0/ --config verbatim --save_path results/loremipsum/llama3.1_70B --port $PORT
python loremipsum/complete_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_completion_v0/ --config exact --save_path results/loremipsum/llama3.1_70B --port $PORT

python loremipsum/complete_lorem.py --ip_path datasets/500/loremipsum/data_bigger.jsonl --prompt_path prompts/loremipsum/zero-shot_completion_v0/ --config verbatim --save_path results/loremipsum/llama3.1_70B --port $PORT
python loremipsum/complete_lorem.py --ip_path datasets/500/loremipsum/data_bigger.jsonl --prompt_path prompts/loremipsum/zero-shot_completion_v0/ --config exact --save_path results/loremipsum/llama3.1_70B --port $PORT

kill $VLLMPID