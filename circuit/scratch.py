import json

def get_data(path):
    """
    Load dataset.
    :param path: dir
    :return: arr
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

data = get_data('corrupted_data/original_after_50.jsonl')
corrupted_data = []
for line in data:
    orig_input = line['input']
    w_idx = orig_input.find('w')
    orig_char = orig_input[w_idx+1]
    if orig_char == '1':
        corrupted_input = orig_input[:w_idx+1] + '0' + orig_input[w_idx+2:]
    elif orig_char == '0':
        corrupted_input = orig_input[:w_idx+1] + '1' + orig_input[w_idx+2:]
    corrupted_data.append({
        "input": corrupted_input,
        'filename': line['filename']
    })

with open("corrupted_data/corrupted_after_50.jsonl", "w") as file:
    for entry in corrupted_data:
        file.write(json.dumps(entry) + "\n")

