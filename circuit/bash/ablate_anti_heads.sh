#!/bin/bash
export HF_TOKEN=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

source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

cd /scratch/yanav/repos/len-gen/circuit/

python ablate_heads.py -m gemma -v instruct -t after -tp anti-induction -l 20