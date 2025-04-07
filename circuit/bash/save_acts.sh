#!/bin/bash
export HF_TOKEN=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

source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

cd /scratch/yanav/repos/len-gen/circuit/

python save_acts.py -m llama -v non-instruct -t after -l 20 -d corrupted
python save_acts.py -m llama -v non-instruct -t after -l 50 -d corrupted
python save_acts.py -m gemma -v instruct -t after -l 20 -d corrupted
python save_acts.py -m gemma -v instruct -t after -l 50 -d corrupted