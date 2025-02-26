PORT=8087

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/scratch/mayank/HF_DATA/
export HF_TOKEN=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

source /nethome/mayank/miniconda3/bin/activate
conda activate /scratch/mayank/envs/mech_interp

export PYTHONPATH=/scratch/mayank/Projects/len-gen

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track

# For OLMo_7B-instruct

vllm serve /scratch/common_models/OLMo-7B-0724-Instruct-hf --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 4096 --max-num-batched-tokens 32000 &

VLLMPID=$!

python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/OLMo_7B-instruct --port $PORT --config "before"
python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/OLMo_7B-instruct --port $PORT --config "before"

python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/OLMo_7B-instruct --port $PORT --config "after"
python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/OLMo_7B-instruct --port $PORT --config "after"

kill $VLLMPID