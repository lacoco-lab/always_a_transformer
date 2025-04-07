import json
import random
import jsonlines

def get_data(path):
    """
    Read jsonl file with the results.

    :param path: dir of the jsonl file
    :return: arr of dicts with results
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def swap_random_even_char_for_w(s: str) -> str:
    # Find valid even indices excluding 0
    valid_even_indices = [i for i in range(2, len(s), 2)]
    if not valid_even_indices:
        return s  # No valid even index to replace

    # Pick one even index at random
    i = random.choice(valid_even_indices)

    # Replace character at that index with 'w'
    s_list = list(s)
    s_list[i] = 'w'
    return ''.join(s_list)

data = get_data("50/flipflop_inductionhead/data.jsonl")
filtered_data = []
for item in data:
    inp = item['input'][:50]
    filtered_data.append({
        'input': swap_random_even_char_for_w(inp),
        'filename': item['filename'],
    })

with jsonlines.open("50/flipflop_inductionhead/data.jsonl", mode="w") as writer:
    writer.write_all(filtered_data)