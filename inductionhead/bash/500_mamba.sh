PORT=8087

export CUDA_VISIBLE_DEVICES=0,1,2,3

source /nethome/mayank/miniconda3/bin/activate
conda activate /scratch/mayank/envs/mech_interp

export PYTHONPATH=/scratch/mayank/Projects/len-gen
#export PYTHONPATH=/home/monk/Projects/len-gen/

export HF_HOME=/scratch/mayank/HF_DATA/
#export HF_HOME=/fast_ssd/HF_DATA/
export HF_TOKEN=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

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track


# For falcon3-mamba-7B-instruct

vllm serve tiiuae/Falcon3-Mamba-7B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-model-len 64000 --max-seq-len-to-capture 128000 --max-num-batched-tokens 128000 &
#vllm serve tiiuae/Falcon3-Mamba-7B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-model-len 32000 --max-seq-len-to-capture 32000 --max-num-batched-tokens 48000 &

VLLMPID=$!

python first_ones/prompt_first_ones.py --ip_path datasets/500/first_or_last/data.jsonl --prompt_path prompts/first_ones/zero-shot_chat_first_v0 --cot "nocot" --save_path results/first_ones/falcon3-mamba-7B-instruct --port $PORT
#python first_ones/prompt_first_ones.py --ip_path datasets/500/first_or_last/data.jsonl --prompt_path prompts/first_ones/zero-shot_chat_first_v0 --cot "cot" --save_path results/first_ones/falcon3-mamba-7B-instruct --port $PORT

python last_ones/prompt_last_ones.py --ip_path datasets/500/first_or_last/data.jsonl --prompt_path prompts/last_ones/zero-shot_chat_last_v0 --cot "nocot" --save_path results/last_ones/falcon3-mamba-7B-instruct --port $PORT
#python last_ones/prompt_last_ones.py --ip_path datasets/500/first_or_last/data.jsonl --prompt_path prompts/last_ones/zero-shot_chat_last_v0 --cot "cot" --save_path results/last_ones/falcon3-mamba-7B-instruct --port $PORT

python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/falcon3-mamba-7B-instruct --port $PORT --config "before"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/falcon3-mamba-7B-instruct --port $PORT --config "before"

python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/falcon3-mamba-7B-instruct --port $PORT --config "after"
#python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/falcon3-mamba-7B-instruct --port $PORT --config "after"

python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "fw-lc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "fw-rc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "lw-lc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "lw-rc"

#python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "fw-lc"
#python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "fw-rc"
#python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "lw-lc"
#python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/falcon3-mamba-7B-instruct --port $PORT --config "lw-rc"

kill $VLLMPID
