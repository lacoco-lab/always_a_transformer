import json
from transformers import AutoTokenizer

def get_data(path):
    """
    Read jsonl file with the results.

    :param path: dir of the jsonl file
    :return: arr of dicts with results
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


data = get_data('../results/loremipsum/OLMo_7B-instruct/zero-shot_chat_v0/500_exact_seed-5.jsonl')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

try:
    max = len(data[0]['tokenized_input'])
    min = len(data[0]['tokenized_input'])
    for inp in data:
        if len(inp['tokenized_input']) > max:
            max = len(inp['tokenized_input'])
        if len(inp['tokenized_input']) < min:
            min = len(inp['tokenized_input'])
except KeyError:
    print("No tokenized input, using HF tokenizer")
    inp_text = tokenizer.tokenize(data[0]['input'])
    max = len(inp_text)
    min = len(inp_text)
    for inp in data:
        inp_text = tokenizer.tokenize(inp['input'])
        if len(inp_text) > max:
            max = len(inp_text)
        if len(inp_text) < min:
            min = len(inp_text)
        
print(max)
print(min)