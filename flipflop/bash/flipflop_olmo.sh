PORT=8087

#new_devices=""
#IFS=',' read -ra my_array <<< "$CUDA_VISIBLE_DEVICES"
#for id in ${my_array[@]};
#do
#    new_devices=${new_devices}`nvidia-smi -L | grep $id | sed -E "s/^GPU ([0-9]+):.*$/\1/"`,
#done
#export CUDA_VISIBLE_DEVICES=${new_devices%?}
#echo "now: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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

python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "fw-lc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "fw-rc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "lw-lc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "nocot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "lw-rc"

python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "fw-lc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "fw-rc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "lw-lc"
python flipflop/prompt_flipflop.py --ip_path datasets/500/flipflop/data.jsonl --prompt_path prompts/flipflop/flipflop_zero-shot_chat_v0 --cot "cot" --save_path results/flipflop/OLMo_7B-instruct --port $PORT --config "lw-rc"

kill $VLLMPID