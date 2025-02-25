from transformers import AutoTokenizer
import json
import os
from collections import defaultdict

path = '../results/last_ones/llama3.1_8B-instruct/qa_zero-shot_chat_last_v0/500_hard_all.jsonl'
output = 'data/last_ones/llama3.1_8b-instruct/qa_last_ones.jsonl'
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

data = []
with open(path, 'r') as f:
    for line in f:
        data.append(json.loads(line))
       
results = []
for line in data:
    full_answer = line['full_answer']
    answer = line['answer']
    gold_ans = line['gold_ans_char']
    
    tokenized_full_answer = tokenizer.tokenize(full_answer)
    cot_ratio = round(len(tokenized_full_answer) / len(line['input']), 2)
    is_correct = str(answer) == str(gold_ans)
    
    stat = {
        "cot_ratio": cot_ratio,
        "is_correct": is_correct
    }
    results.append(stat)
    
grouped = defaultdict(lambda: {'correct': 0, 'total': 0})

for d in results:
    grouped[d['cot_ratio']]['correct'] += d['is_correct']
    grouped[d['cot_ratio']]['total'] += 1

accuracy_by_cot_ratio = {cot_ratio: values['correct'] / values['total'] for cot_ratio, values in grouped.items()}
di = []
for cot_ratio, accuracy in accuracy_by_cot_ratio.items():
    di.append({"cot_ratio": cot_ratio, "accuracy": round(accuracy, 2)})

if not os.path.exists(output):
    with open(output, "w") as file:
        for entry in di:
            file.write(json.dumps(entry) + "\n")
    print(f"File '{output}' created and data written.")
else:
    print(f"File '{output}' already exists. No changes made.")