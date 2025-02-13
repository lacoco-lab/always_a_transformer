import json
import time
import re
from together import Together
from prompt_utils import render_prompt

client = Together()
system_path = "system.jinja"
task_path = "templates/flipflop-traverse-left.jinja"
output_file = "results/flipflop-traverse-left/flipflop_500_pw0.1.jsonl"
flipflop_data = "../results/flipflop/llama3.3_70B-instruct/sparse/s1/flipflop_500_pw0_results.jsonl"

queries = []
with open(flipflop_data, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        system_prompt, task_prompt = render_prompt(system_path, task_path, data['flipflop'])
        queries.append(
            {
                'task_prompt': task_prompt,
                'flipflop': data['flipflop'],
                'last_valid_token': data['last_valid_token']
            }
        )

answers = []
query_count = 0
for query in queries:
    if query_count >= 6:
        print("Submitted 6 queries... taking a break.")
        time.sleep(10)  # API limit = 6 queries per minute
        query_count = 0

    response_text = ""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": task_prompt
            },
        ],
        max_tokens=None,
        temperature=0,
        top_p=0.7,
        top_k=1,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True
    )

    for token in response:
        if hasattr(token, 'choices') and token.choices[0].delta.content:
            response_text += token.choices[0].delta.content

    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    extracted_answer = match.group(1) if match else response_text
    
    query['response'] = response_text
    query['answer'] = extracted_answer
    
    answers.append(query)
    query_count += 1
    
with open(output_file, 'w', encoding='utf-8') as file:
    for answer in answers:
        file.write(json.dumps(answer) + "\n")

print("Finished processing.")