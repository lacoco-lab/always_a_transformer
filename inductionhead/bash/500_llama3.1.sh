PORT=8087

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#
#source /nethome/mayank/miniconda3/bin/activate
#conda activate /scratch/mayank/envs/mech_interp

#export PYTHONPATH=/scratch/mayank/Projects/len-gen

#export VLLM_NO_USAGE_STATS=1
#export DO_NOT_TRACK=1
#mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track


# For llama3.1_8B-instruct

#vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 128000 &
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 32000 &

VLLMPID=$!

python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/llama3.1_8B-instruct --port $PORT --config "before"
python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/llama3.1_8B-instruct --port $PORT --config "before"

python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop_inductionhead/llama3.1_8B-instruct --port $PORT --config "after"
python inductionhead/prompt_inductionhead.py --ip_path datasets/500/flipflop_inductionhead/data.jsonl --prompt_path prompts/flipflop_inductionhead/inductionhead_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop_inductionhead/llama3.1_8B-instruct --port $PORT --config "after"

kill $VLLMPID