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


def clean_tokens(tokens):
    """
    Clean up trailing spaces and 'Ġ'
    :param tokens: arr, tokenized input/output
    :return: arr, cleaned input/output
    """
    
    cleaned = []
    for tok in tokens:
        tok = tok.strip().replace('Ġ', '')
        cleaned.append(tok)
    return cleaned


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
    incorrect_firsts = []
    correct = 0
    for gold_ans, ans in zip(inputs, outputs):
        if ans[0] == gold_ans[0]:
            correct += 1
        else:
            if ans[0] == '': # olmo tokenization issue
                if ans[1] == gold_ans[0]:
                    correct += 1 
                    continue
            incorrect_firsts.append({'input': gold_ans, 'output': ans, 'token': gold_ans[0]})
    
    return correct / len(inputs), incorrect_firsts


def get_accuracy_last_str(inputs, outputs):
    """
    OLMo has a lot of tokenization issue. This is a sanity check to check whether the last token in the
    response string matches (separated by a space, not processed by a tokenizer)
    
    :param inputs: arr[str], gold answer
    :param outputs: arr[str], answer
    :return: float, accuracy of the last token copy
    """
    
    correct = 0
    for gold_ans, ans in zip(inputs, outputs):
        split_ans = ans.split()
        split_gold_ans = gold_ans.split()
        if split_ans[len(split_ans)-1].strip('.') == split_gold_ans[len(split_gold_ans)-1].strip('.'):
            correct += 1
    return correct / len(inputs)


def get_accuracy_last(inputs, outputs):
    """
    Calculate accuracy of the last token for the task.
    
    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the last token copy
    """
    incorrect_lasts = []
    correct = 0
    for ans, gold_ans in zip(inputs, outputs):
        if ans[len(ans)-1] == gold_ans[len(gold_ans)-1]:
            correct += 1
        else:
            incorrect_lasts.append({'input': gold_ans, 'output': ans, 'token': gold_ans[len(gold_ans)-1]})

    return correct / len(inputs), incorrect_lasts


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
    faulty_copies = []
    
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
                    else:
                        #print(f'Token {token} was misplaced from {gold_ans.index(token)} pos to {ans.index(token)} pos.')
                        faulty_copies.append({'input': gold_ans, 'output': ans, 'token': token, 'pos': gold_ans.index(token),
                                              'mis_pos': ans.index(token)})
                        break
                except ValueError: # Token has not been copied at all
                    absent_copies.append({'input': gold_ans, 'output': ans, 'token': token})
                    #print(f'Token {token} has not been copied correctly.')
                    
    return correct / total_unique, absent_copies, faulty_copies, total_unique


def get_accuracy_repeats(inputs, outputs):
    """
    Calculate how accurately consecutive repeated tokens are copied
    from the input to the output in the same positions.

    :param inputs: list of lists (tokenized inputs)
    :param outputs: list of lists (tokenized outputs)
    :return: float  -> accuracy of repeated-token copy
             list   -> absent copies (repeated pairs missing entirely in output)
             list   -> faulty copies (repeated pairs present but in the wrong position)
             int    -> total number of repeated pairs found in inputs
    """

    correct = 0
    total_repeated = 0

    absent_copies = []
    faulty_copies = []

    for gold_ans, ans in zip(inputs, outputs):

        for i in range(len(gold_ans) - 1):
            if gold_ans[i] == gold_ans[i + 1]:
                repeated_token = gold_ans[i]
                total_repeated += 1

                if i < len(ans) - 1:
                    if ans[i] == repeated_token and ans[i + 1] == repeated_token:
                        correct += 1
                    else:
                        found_pair_elsewhere = False
                        for j in range(len(ans) - 1):
                            if ans[j] == repeated_token and ans[j + 1] == repeated_token:
                                # Found it in a different position
                                faulty_copies.append({
                                    'input': gold_ans,
                                    'output': ans,
                                    'token': repeated_token,
                                    'pos': i,
                                    'mis_pos': j
                                })
                                found_pair_elsewhere = True
                                break
                        if not found_pair_elsewhere:
                            absent_copies.append({
                                'input': gold_ans,
                                'output': ans,
                                'token': repeated_token,
                                'pos': i
                            })
                else:
                    absent_copies.append({
                        'input': gold_ans,
                        'output': ans,
                        'token': repeated_token,
                        'pos': i
                    })

    accuracy = correct / total_repeated if total_repeated else 0.0

    return accuracy, absent_copies, faulty_copies, total_repeated


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
OUTPUT_FILE = f"output_{model}_{args.prompt_type}_{args.seed}.txt"

# select correct slices
if args.model == 'olmo':
    start_inp, end_inp = 28, 41
    start_out, end_out = 3, 8
    if args.seed == 'VERBATIM':
        start_inp, end_inp = 34, 41
        start_out, end_out = 3, 8
elif 'llama' in args.model:
    start_inp, end_inp = 72, 37
    start_out, end_out = 3, 6
    if args.seed == 'VERBATIM':
        start_inp, end_inp = 78, 37
        start_out, end_out = 3, 6
        
new_data = []
# Fix the issue of multiple repetitions in verbatim seed 
if model == 'llama3.1_8B-instruct' and args.seed == 'VERBATIM':
    for line in data:
        if len(line['answer'].split('\n')) > 1:
            line['answer'] = line['answer'].split('\n')[1]
        new_data.append(line)
        
if new_data != []:
    data = new_data

inputs, outputs = [], []
str_inputs, str_outputs = [], []
incorrect_outputs = []
for line in data:
    cleaned_inputs = clean_tokens(line['tokenized_input'][start_inp:len(line['tokenized_input'])-end_inp])
    cleaned_outputs = clean_tokens(line['tokenized_output'][start_out:len(line['tokenized_output'])-end_out])
    str_inputs.append(line['gold_ans'])
    str_outputs.append(line['answer'])
    
    if line['is_correct'] == False:
        incorrect_outputs.append({'tokenized_input': cleaned_inputs, 'tokenized_output': cleaned_outputs})
    
    inputs.append(cleaned_inputs)
    outputs.append(cleaned_outputs)

accuracy_first, incorrect_firsts = get_accuracy_first(inputs, outputs)
accuracy_last, incorrect_lasts = get_accuracy_last(inputs, outputs)
accuracy_ind, absent_copies, faulty_copies, total_unique = get_accuracy_ind(inputs, outputs)
accuracy_last_str = get_accuracy_last_str(str_inputs, str_outputs)
accuracy_repeated, absent_repeated, faulty_repeated, total_repeated =  get_accuracy_repeats(inputs, outputs)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    print(f'First accuracy: {accuracy_first*100}%\n')
    print(f'Last accuracy: {accuracy_last*100}%\n')
    print(f'Last accuracy (string manner): {accuracy_last_str*100}%\n')
    print(f'Induction accuracy: {accuracy_ind*100}% for total of {total_unique} tokens.\n')
    print(f'Repeating tokens accuracy: {accuracy_repeated*100}%\n')
    
    f.write(f'First accuracy: {accuracy_first*100}%\n')
    if len(incorrect_firsts) > 0:
        f.write(f'Incorrect first copies: {len(incorrect_firsts)}\n')
        for inp in incorrect_firsts:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")

    f.write(f'Last accuracy: {accuracy_last*100}%\n')
    if len(incorrect_lasts) > 0:
        f.write(f'Incorrect last copies: {len(incorrect_lasts)}\n')
        for inp in incorrect_lasts:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")
    f.write(f'Induction accuracy: {accuracy_ind*100}% for total of {total_unique} tokens.\n')
    

    if len(absent_copies) > 0:
        f.write(f'Absent copies: {len(absent_copies)}\n')
        for inp in absent_copies:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")

    if len(faulty_copies) > 0:
        f.write(f'Faulty copies: {len(faulty_copies)}\n')
        for inp in faulty_copies:
            f.write(str(inp['token']) + " from pos " + str(inp['pos']) + " to " + str(inp['mis_pos']) + "\n" + str(
                inp['input']) + "\n" + str(inp['output']) + "\n" + "=====================\n")

    f.write(f'Repeated tokens accuracy: {accuracy_repeated * 100}%\n')

    if len(absent_repeated) > 0:
        f.write(f'Absent repeated copies: {len(absent_repeated)}\n')
        for inp in absent_repeated:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")

    if len(faulty_repeated) > 0:
        f.write(f'Faulty copies: {len(faulty_repeated)}\n')
        for inp in faulty_repeated:
            f.write(str(inp['token']) + " from pos " + str(inp['pos']) + " to " + str(inp['mis_pos']) + "\n" + str(
                inp['input']) + "\n" + str(inp['output']) + "\n" + "=====================\n")
            
    if len(incorrect_outputs) > 0:
        f.write(f'Incorrect outputs: {len(incorrect_outputs)} (may overlap with other mistakes)\n')
        for inp in incorrect_outputs:
            f.write(str(inp['tokenized_input']) + "\n" + str(inp['tokenized_output'])+ "\n" + "=====================\n")

print(f"Output written to {OUTPUT_FILE}")