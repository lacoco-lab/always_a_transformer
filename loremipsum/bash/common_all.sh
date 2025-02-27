PORT=8087

export CUDA_VISIBLE_DEVICES=0,1,2,3

source /nethome/mayank/miniconda3/bin/activate
conda activate /scratch/mayank/envs/mech_interp

export PYTHONPATH=/scratch/mayank/Projects/len-gen

export HF_HOME=/scratch/mayank/HF_DATA/
export HF_TOKEN=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

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track


# For llama3.3_70B-instruct

#vllm serve /scratch/common_models/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 32000 &
#
#VLLMPID=$!
#
#python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "exact"
#python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.3_70B-instruct --port $PORT --config "verbatim"
#
#kill $VLLMPID
#
#sleep 5

# For llama3.1_8B-instruct

vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 32000 --max-num-batched-tokens 32000 &

VLLMPID=$!

python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.1_8B-instruct --port $PORT --config "exact"
python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/llama3.1_8B-instruct --port $PORT --config "verbatim"

kill $VLLMPID

sleep 5

# For OLMo_7B-instruct

vllm serve /scratch/common_models/OLMo-7B-0724-Instruct-hf --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-seq-len-to-capture 4096 --max-num-batched-tokens 32000 &

VLLMPID=$!

python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/OLMo_7B-instruct --port $PORT --config "exact"
python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/OLMo_7B-instruct --port $PORT --config "verbatim"

kill $VLLMPID

sleep 5

# For falcon3-mamba-7B-instruct

vllm serve tiiuae/Falcon3-Mamba-7B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --disable-log-stats --seed 5 --api-key "sk_noreq" --host 0.0.0.0 --port $PORT --max-model-len 64000 --max-seq-len-to-capture 128000 --max-num-batched-tokens 128000 &

VLLMPID=$!

python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/falcon3-mamba-7B-instruct --port $PORT --config "exact"
python loremipsum/prompt_lorem.py --ip_path datasets/500/loremipsum/data.jsonl --prompt_path prompts/loremipsum/zero-shot_chat_v0 --save_path results/loremipsum/falcon3-mamba-7B-instruct --port $PORT --config "verbatim"

kill $VLLMPID