#!/bin/bash
export HF_TOKEN=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

source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

cd /scratch/yanav/repos/len-gen/circuit/

python ablate_heads.py -m gemma -v instruct -t after -tp induction -l 20
python ablate_heads.py -m gemma -v instruct -t after -tp induction -l 50
python ablate_heads.py -m gemma -v instruct -t after -tp random-all -l 20
python ablate_heads.py -m gemma -v instruct -t after -tp random-all -l 50