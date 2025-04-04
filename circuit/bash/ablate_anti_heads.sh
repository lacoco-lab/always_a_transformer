#!/bin/bash
export HF_TOKEN=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

source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

cd /scratch/yanav/repos/len-gen/circuit/

python ablate_heads.py -m llama -v instruct -t after -tp induction -l 20
python ablate_heads.py -m llama -v instruct -t after -tp induction -l 50
python ablate_heads.py -m llama -v instruct -t after -tp random-beg -l 20
python ablate_heads.py -m llama -v instruct -t after -tp random-beg -l 50
python ablate_heads.py -m llama -v instruct -t after -tp random-mid -l 20
python ablate_heads.py -m llama -v instruct -t after -tp random-mid -l 50
python ablate_heads.py -m llama -v instruct -t after -tp random-end -l 20
python ablate_heads.py -m llama -v instruct -t after -tp random-end -l 50
python ablate_heads.py -m llama -v instruct -t after -tp random-all -l 20
python ablate_heads.py -m llama -v instruct -t after -tp random-all -l 50