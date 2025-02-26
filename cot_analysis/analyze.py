import json
import os
from collections import defaultdict

base_folder = '../results/flipflop_inductionhead/llama3.3_70b-instruct/inductionhead_zero-shot_chat_v0'
output = 'data/flipflop/llama3.3_70b-instruct/flipflop_ind_after.jsonl'

data = []

for file in os.listdir(base_folder):
    if file.endswith(".jsonl") and 'after' in file:
        print(file)
        file_path = os.path.join(base_folder, file)

        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line)) 
        data = data[:-100]

print(f"Total records loaded: {len(data)}")

       
results = []
for line in data:
    output_length = line['output_length']
    answer = line['answer']
    gold_ans = line['gold_ans_char']
    
    cot_ratio = round(output_length / len(line['input']), 2)
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