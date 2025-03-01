import json
from argparse import ArgumentParser
import os

"""
Args options:

Seed:
- EXACT
- VERBATIM
- BIGGER

Models:
- llama_8b
- llama_70b
- olmo
"""

EXACT_SEED = '500_exact_seed-5.jsonl'
VERBATIM_SEED = '500_verbatim_seed-5.jsonl'
BIGGER_SEED = 'bigger_exact_seed-5.jsonl'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = 'loremipsum'
RESULTS = 'results'

def get_data(path):
    """
    Read jsonl file with the results.
    
    :param path: dir of the jsonl file
    :return: arr of dicts with results
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def get_accuracy_first(inputs, outputs):
    """
    Calculate accuracy of the first token for the task.
    
    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the first token copy
    """
    
    correct = 0
    for ans, gold_ans in zip(inputs, outputs):
        if ans[0] == gold_ans[0]:
            correct += 1
    
    return correct / len(inputs)


def get_accuracy_last(inputs, outputs):
    """
    Calculate accuracy of the last token for the task.
    
    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the last token copy
    """

    correct = 0
    for ans, gold_ans in zip(inputs, outputs):
        if ans[len(ans)-1] == gold_ans[len(gold_ans)-1]:
            correct += 1

    return correct / len(inputs)


def get_accuracy_ind(inputs, outputs):
    """
    Calculate whether a unique token was copied correctly.

    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the unique token copy; arr of examples of token not copied
    """
    
    correct = 0 
    total_unique = 0
    
    absent_copies = []
    
    for gold_ans, ans in zip(inputs, outputs):
        counts = {}
        for tok in gold_ans:
            counts[tok] = counts.get(tok, 0) + 1
            
        unique_tokens = [k for k, v in counts.items() if v == 1]
        total_unique += len(unique_tokens)
        
        if len(unique_tokens) != 0:
            for token in unique_tokens:
                try:
                    if ans.index(token) == gold_ans.index(token):
                        correct += 1
                except ValueError: # Token has not been copied at all
                    absent_copies.append({'input': gold_ans, 'output': ans, 'token': token})
                    print(f'No token {token} has been copied correctly.')
                    
    return correct / total_unique, absent_copies

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                        help="choose model to parse results from")
parser.add_argument("-s", "--seed", dest="seed",
                        help="choose seed to load")
parser.add_argument("-pt", "--prompt_type", dest="prompt_type",
                    help="choose prompt type folder", default="zero-shot_chat_v0")

args = parser.parse_args()

seed_path = ''
if args.seed == 'EXACT': 
    seed_path = EXACT_SEED
elif args.seed == 'VERBATIM': 
    seed_path = VERBATIM_SEED
elif args.seed == 'BIGGER':
    seed_path = BIGGER_SEED
else:
    raise ValueError(f'Seed {args.seed} not recognized')

model = ''
if args.model == 'llama_8b':
    model = "llama3.1_8B-instruct"
elif args.model == 'llama_70b':
    model = "llama3.3_70B-instruct"
elif args.model == 'olmo':
    model = "OLMo_7B-instruct"
else:
    raise ValueError(f'Model {args.model} not recognized')

path = os.path.join(ROOT, RESULTS,  TASK, model, args.prompt_type, seed_path)
data = get_data(path)

inputs, outputs = [], []
for line in data:
    inputs.append(line['input'])
    outputs.append(line['answer'])

accuracy_first = get_accuracy_first(inputs, outputs)
accuracy_last = get_accuracy_last(inputs, outputs)
accuracy_ind, absent_copies = get_accuracy_ind(inputs, outputs)

print(f'First accuracy: {accuracy_first*100}%\n')
print(f'Last accuracy: {accuracy_last*100}%\n')
print(f'Induction accuracy: {accuracy_ind*100}%\n')
if len(absent_copies) > 0:
    print(f'Absent copies: {len(absent_copies)}')
    for inp in absent_copies:
        print(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(inp['output']) + "\n" + "=====================")
