#!/bin/bash
export HF_TOKEN=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

source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

cd /scratch/yanav/repos/len-gen/circuit/

python ablate_heads.py -m llama -v non-instruct -t before -tp anti-induction
python ablate_heads.py -m llama -v non-instruct -t after -tp anti-induction
python ablate_heads.py -m llama -v instruct -t before -tp anti-induction
python ablate_heads.py -m llama -v instruct -t after -tp anti-induction
python ablate_heads.py -m pythia -v non-instruct -t before -tp anti-induction
python ablate_heads.py -m pythia -v non-instruct -t after -tp anti-induction