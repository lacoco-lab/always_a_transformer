PORT=8087

python last_ones/prompt_last_ones.py --ip_path datasets/last_ones/500_hard_all.jsonl --prompt_path prompts/last_ones/zero-shot_chat_last/ --save_path results/last_ones/llama3.3_70B-instruct --port $PORT
#python last_ones/prompt_last_ones.py --ip_path datasets/last_ones/500_hard_all.jsonl --prompt_path prompts/last_ones/zero-shot_chat_s_last/ --save_path results/s_last_ones/llama3.3_70B-instruct --port $PORT

python first_ones/prompt_first_ones.py --ip_path datasets/last_ones/500_hard_all.jsonl --prompt_path prompts/first_ones/zero-shot_chat_first_v0/ --save_path results/first_ones/llama3.3_70B-instruct --port $PORT
#python first_ones/prompt_first_ones.py --ip_path datasets/last_ones/500_hard_all.jsonl --prompt_path prompts/first_ones/zero-shot_chat_s_first_v0/ --save_path results/s_first_ones/llama3.3_70B-instruct --port $PORT