#!/bin/bash
export HF_TOKEN=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

source /scratch/yanav/anaconda3/bin/activate
conda activate len-gen

cd /scratch/yanav/repos/len-gen/circuit/

python ablate_with_zeroes.py -m gemma -v instruct -t after -tp high_att_l20 -l 50
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp high_att_l50 -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp high_att_l20 -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp high_att_l50 -l 50
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp induction -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-beg -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-mid -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-end -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-all -l 20
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp induction -l 50
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-beg -l 50
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-end -l 50
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-mid -l 50
python ablate_with_zeroes.py -m gemma -v instruct -t after -tp random-all -l 50