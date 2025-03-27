PORT=8091

if [ $HOSTNAME == "toruk" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    export HF_HOME=/fast_ssd/HF_DATA/
    TP_SIZE=2
    GPU_MEM=0.85
else
    export CUDA_VISIBLE_DEVICES=0,1
    source /nethome/mayank/miniconda3/bin/activate
    conda activate /scratch/mayank/envs/t3
    
    export PYTHONPATH=/scratch/mayank/Projects/len-gen
    export HF_HOME=/scratch/mayank/HF_DATA/
    TP_SIZE=2
    GPU_MEM=0.95
fi

export HF_TOKEN=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

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track



#vllm serve /scratch/common_models/QwQ-32B --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 256000 &
#
#VLLMPID=$!

#python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/QwQ-32B --port $PORT --config "before"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/QwQ-32B --port $PORT --config "before"

#python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/QwQ-32B --port $PORT --config "after"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/QwQ-32B --port $PORT --config "after"

#kill $VLLMPID

#sleep 5

# For Meta-Llama-3-8B-Instruct
#
#CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --disable-log-stats --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 256 --max-num-seqs 64 --max-num-batched-tokens 32000 --no-enable-prefix-caching &
#
#VLLMPID=$!
#
##python inductionhead/prompt_inductionhead.py --ip_path datasets/100/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "before" 
##python inductionhead/prompt_inductionhead.py --ip_path datasets/100/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "before"
#
##python inductionhead/prompt_inductionhead.py --ip_path datasets/100/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "after"
##python inductionhead/prompt_inductionhead.py --ip_path datasets/100/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "after"
#
#python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/Meta-Llama-3-8B-Instruct --port $PORT --config "before"
##python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "before"
#
#python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/Meta-Llama-3-8B-Instruct --port $PORT --config "after"
##python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "after"
#
#python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/Meta-Llama-3-8B-Instruct --port $PORT --config "before"
##python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "before"
#
#python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/Meta-Llama-3-8B-Instruct --port $PORT --config "after"
##python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/Meta-Llama-3-8B-Instruct --port $PORT --config "after"
#
#kill $VLLMPID
#
#sleep 5

# For google/gemma-2-9b-it

CUDA_VISIBLE_DEVICES=0,1 vllm serve google/gemma-2-9b-it --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --disable-log-stats --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 512 --max-num-seqs 64 --max-num-batched-tokens 32000 --no-enable-prefix-caching &

VLLMPID=$!

python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/gemma-2-9b-it --port $PORT --config "before"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/gemma-2-9b-it --port $PORT --config "before"

python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/gemma-2-9b-it --port $PORT --config "after"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/50/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/gemma-2-9b-it --port $PORT --config "after"

python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/gemma-2-9b-it --port $PORT --config "before"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/gemma-2-9b-it --port $PORT --config "before"

python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead_rev_digit/gemma-2-9b-it --port $PORT --config "after"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/20/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/gemma-2-9b-it --port $PORT --config "after"

kill $VLLMPID

sleep 5