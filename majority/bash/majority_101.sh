#!/bin/bash

# we need to rename gpus in order to access them via CUDA_VISIBLE_DEVICES

new_devices=""
IFS=',' read -ra my_array <<< "$CUDA_VISIBLE_DEVICES"
for id in ${my_array[@]};
do
    new_devices=${new_devices}`nvidia-smi -L | grep $id | sed -E "s/^GPU ([0-9]+):.*$/\1/"`,
done
export CUDA_VISIBLE_DEVICES=${new_devices%?}
echo "now: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source /nethome/mayank/miniconda3/bin/activate
conda activate /scratch/mayank/envs/mech_interp

export HF_HOME=/scratch/mayank/HF_DATA/
export PYTHONPATH=/scratch/mayank/Projects/len-gen
export MKL_SERVICE_FORCE_INTEL=1

cd /scratch/mayank/Projects/len-gen
python majority/prompt_majority.py --ip_path datasets/majority/ --prompt_path prompts/majority/zero-shot_chat_tie1/ --save_path results/majority/ --model /scratch/common_models/Llama-3.3-70B-Instruct
python majority/prompt_majority.py --ip_path datasets/majority/ --prompt_path prompts/majority/zero-shot_chat_tie0/ --save_path results/majority/ --model /scratch/common_models/Llama-3.3-70B-Instruct