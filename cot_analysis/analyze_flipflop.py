from transformers import AutoTokenizer
import json
import os
from collections import defaultdict

base_folder = '../results/flipflop/llama3.3_70B-instruct/sparse-qa'
output = 'data/flipflop/llama3.3_70b-instruct/sparse-qa.jsonl'
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')

all_data = []

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.endswith(".jsonl"):
                print(file)
                file_path = os.path.join(subfolder_path, file)

                with open(file_path, "r") as f:
                    for line in f:
                        all_data.append(json.loads(line)) 

print(f"Total records loaded: {len(all_data)}")


results = []
for line in all_data:
    full_answer = line['full_answer']
    answer = line['answer']
    gold_ans = line['last_valid_token']

    tokenized_full_answer = tokenizer.tokenize(full_answer)
    cot_ratio = round(len(tokenized_full_answer) / len(line['prompt']), 2)
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
print(di)

if not os.path.exists(output):
    with open(output, "w") as file:
        for entry in di:
            file.write(json.dumps(entry) + "\n")
    print(f"File '{output}' created and data written.")
else:
    print(f"File '{output}' already exists. No changes made.")