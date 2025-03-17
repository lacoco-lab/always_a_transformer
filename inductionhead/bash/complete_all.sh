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


## For llama3.1_70
#
#vllm serve /scratch/common_models/Llama-3.1-70B/ --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 256000 &
#
#VLLMPID=$!
#
#python inductionhead/complete_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0/ --config 'before' --save_path results/flipflop_inductionhead/llama3.1_70B --port $PORT
#python inductionhead/complete_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0/ --config 'after' --save_path results/flipflop_inductionhead/llama3.1_70B --port $PORT
#
#kill $VLLMPID
#
#sleep 5
#
## For llama3.1_8B
#
#vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 256000 &
#
#VLLMPID=$!
#
#python inductionhead/complete_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0/ --config 'before' --save_path results/flipflop_inductionhead/llama3.1_8B --port $PORT
#python inductionhead/complete_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0/ --config 'after' --save_path results/flipflop_inductionhead/llama3.1_8B --port $PORT
#
#kill $VLLMPID
#
#sleep 5
#
## For OLMo_7B
#
#vllm serve /scratch/common_models/OLMo-7B-0724-hf --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 4096 --max-num-batched-tokens 256000 &
#
#VLLMPID=$!
#
#python inductionhead/complete_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0/ --config 'before' --save_path results/flipflop_inductionhead/OLMo_7B --port $PORT
#python inductionhead/complete_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0/ --config 'after' --save_path results/flipflop_inductionhead/OLMo_7B --port $PORT
#
#kill $VLLMPID


vllm serve EleutherAI/pythia-1.4b-deduped --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --disable-log-stats --api-key "sk_noreq" --host 0.0.0.0 --port 8087 --max-seq-len-to-capture 500 --max-num-batched-tokens 32000 &

VLLMPID=$!

python inductionhead/complete_inductionhead.py --ip_path datasets/100/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0 --config 'before' --save_path results/flipflop_inductionhead/pythia-1.4b --port $PORT
python inductionhead/complete_inductionhead.py --ip_path datasets/100/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_completion_v0 --config 'after' --save_path results/flipflop_inductionhead/pythia-1.4b --port $PORT

kill $VLLMPID