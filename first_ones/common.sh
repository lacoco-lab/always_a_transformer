PORT=8087
python first_ones/prompt_first_ones.py --ip_path datasets/last_ones/500_hard_all.jsonl --prompt_path prompts/first_ones/zero-shot_chat_first_v0/ --save_path results/first_ones/llama3.3_70B-instruct --port $PORT
