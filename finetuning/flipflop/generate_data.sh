# first
python finetuning/flipflop/generate_data.py -min 3 -max 50 -n 50000 -c first_right -o datasets/fine_tuning/flipflop/first_right_train.jsonl -s 51
python finetuning/flipflop/generate_data.py -min 51 -max 100 -n 5000 -c first_right -o datasets/fine_tuning/flipflop/first_right_test_ood.jsonl -s 51

python finetuning/flipflop/generate_data.py -min 3 -max 50 -n 50000 -c first_left -o datasets/fine_tuning/flipflop/first_left_train.jsonl -s 51
python finetuning/flipflop/generate_data.py -min 51 -max 100 -n 5000 -c first_left -o datasets/fine_tuning/flipflop/first_left_test_ood.jsonl -s 51

#last
python finetuning/flipflop/generate_data.py -min 3 -max 50 -n 50000 -c last_right -o datasets/fine_tuning/flipflop/last_right_train.jsonl -s 51
python finetuning/flipflop/generate_data.py -min 51 -max 100 -n 5000 -c last_right -o datasets/fine_tuning/flipflop/last_right_test_ood.jsonl -s 51

python finetuning/flipflop/generate_data.py -min 3 -max 50 -n 50000 -c last_left -o datasets/fine_tuning/flipflop/last_left_train.jsonl -s 51
python finetuning/flipflop/generate_data.py -min 51 -max 100 -n 5000 -c last_left -o datasets/fine_tuning/flipflop/last_left_test_ood.jsonl -s 51