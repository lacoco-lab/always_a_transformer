import json
from argparse import ArgumentParser
import os
import re

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

EXACT_SEED = '4000_exact_seed-5.jsonl'
VERBATIM_SEED = '500_verbatim_seed-5.jsonl'
BIGGER_SEED = 'bigger_verbatim_seed-5.jsonl'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = 'loremipsum'
RESULTS = 'results'




def clean(text):
    """
    Clean and split the input and answer string by whitespace, remove all punctiation marks and characters
    :param text: string
    :return: arr of strings of cleaned text
    """
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    words = cleaned_text.split()
    return words


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
        try:
            if ans[0] == gold_ans[0]:
                correct += 1
            else:
                if ans[0] == '': # olmo tokenization issue
                    if ans[1] == gold_ans[0]:
                        correct += 1 
                        continue
                incorrect_firsts.append({'input': gold_ans, 'output': ans, 'token': gold_ans[0]})
        except IndexError:
            continue
            #print(ans)
    return correct / len(inputs), incorrect_firsts


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
        try:
            if ans[len(ans)-1] == gold_ans[len(gold_ans)-1]:
                correct += 1
            else:
                incorrect_lasts.append({'input': gold_ans, 'output': ans, 'token': gold_ans[len(gold_ans)-1]})
        except IndexError:
            continue
            #print(ans)
    return correct / len(inputs), incorrect_lasts


def get_accuracy_repeated_bigrams(inputs, outputs):
    """
    Calculate accuracies of copies of all unique bigrams in the paragraph.

    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the unique bigrams, total unique bigrams
    """

    correct = 0
    total = 0
    absent_copies = []

    for gold_ans, ans in zip(inputs, outputs):
        all_ans_bigrams = []
        for i in range(len(ans) - 1):
            all_ans_bigrams.append((ans[i], ans[i + 1]))

        # Get all non-deterministic bigrams
        bigrams = {}
        unique_toks = set(gold_ans)
        for tok in unique_toks:
            if not tok.isalpha():
                continue
            all_tok_bigrams_idx = [i for i, val in enumerate(gold_ans) if val == tok and i < len(gold_ans) - 1]
            all_tok_bigrams = []
            for idx in all_tok_bigrams_idx:
                if gold_ans[idx+1].isalpha():
                    bigram = (gold_ans[idx], gold_ans[idx + 1])
                    all_tok_bigrams.append(bigram)
            bigrams[tok] = set(all_tok_bigrams)
        repeated_bigrams = []
        for k in bigrams.keys():
            if len(bigrams[k]) > 1:
                repeated_bigrams.append(list(bigrams[k]))
                
        # flatten the list
        repeated_bigrams = [x for xs in repeated_bigrams for x in xs]

        for bigram in repeated_bigrams:
            if bigram in all_ans_bigrams:
                correct += 1
            else:
                absent_copies.append({'token': bigram, 'input': gold_ans, 'output': ans})
                #print(f'Failed to copy a non-deterministic bigram {bigram}')
            total += 1

    return correct / total, absent_copies, total


def get_accuracy_unique_bigrams(inputs, outputs):
    """
    Calculate accuracies of copies of all unique bigrams in the paragraph.
    
    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the unique bigrams, total unique bigrams
    """
    
    correct = 0
    total = 0
    absent_copies = []
    
    for gold_ans, ans in zip(inputs, outputs):
        all_ans_bigrams = []
        for i in range(len(ans) - 1):
            all_ans_bigrams.append((ans[i], ans[i + 1]))
        
        # Get all unique bigrams
        bigrams = {}
        unique_toks = set(gold_ans)
        for tok in unique_toks:
            if not tok.isalpha():
                continue
            all_tok_bigrams_idx = [i for i, val in enumerate(gold_ans) if val == tok and i < len(gold_ans)-1]
            all_tok_bigrams  = []
            for idx in all_tok_bigrams_idx:
                if gold_ans[idx + 1].isalpha():
                    bigram = (gold_ans[idx], gold_ans[idx+1])
                    all_tok_bigrams.append(bigram)
            bigrams[tok] = set(all_tok_bigrams)
        unique_bigrams = []
        for k in bigrams.keys():
            if len(bigrams[k]) == 1:
                unique_bigrams.append(list(bigrams[k])[0])
        
        for bigram in unique_bigrams:
            if bigram in all_ans_bigrams:
                correct += 1
            else:
                absent_copies.append({'token': bigram, 'input': gold_ans, 'output': ans})
                #print(f'Failed to copy a unique bigram {bigram}')
            total += 1
    
    return correct / total, absent_copies, total
        

def get_accuracy_unique_right(inputs, outputs):
    """
    Calculate whether a unique token with its right token was copied correctl

    :param inputs: arr of tokenized inputs
    :param outputs: arr of tokenized outputs
    :return: float, accuracy of the unique token copy; arr of examples of token not copied, total unique tokens
    """
    
    correct_after = 0
    total_unique = 0
    absent_copies = []
    
    for gold_ans, ans in zip(inputs, outputs):
        
        counts = {}
        for tok in gold_ans:
            counts[tok] = counts.get(tok, 0) + 1
            
        unique_tokens = [k for k, v in counts.items() if v == 1]
        
        for tok in unique_tokens:
            if not tok.isalpha():
                continue
            unique_pos_gold = gold_ans.index(tok)
            
            try:
                unique_pos_ans = ans.index(tok)
            except ValueError:
                #print(f"Failed to copy the unique token {tok}")
                absent_copies.append({'input': gold_ans, 'output': ans, 'token': tok})
                continue
            
            if gold_ans[unique_pos_gold:unique_pos_gold+2] == ans[unique_pos_ans:unique_pos_ans+2]:
                correct_after += 1
            
            total_unique += 1
                    
    return correct_after/total_unique, absent_copies, total_unique



parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                        help="choose model to parse results from")
parser.add_argument("-s", "--seed", dest="seed",
                        help="choose seed to load")
parser.add_argument("-pt", "--prompt_type", dest="prompt_type",
                    help="choose prompt type folder", default="zero-shot_chat_v0")
parser.add_argument("-v", "--version", dest="version", help="instruct or completion model version")

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
    if args.version == 'instruct':
        model = "llama3.3_70B-instruct"
    elif args.version == 'completion':
        model = "llama3.1_70B"
elif args.model == 'olmo':
    model = "OLMo_7B-instruct"
else:
    raise ValueError(f'Model {args.model} not recognized')

path = os.path.join(ROOT, RESULTS,  TASK, model, args.prompt_type, seed_path)
data = get_data(path)
OUTPUT_FILE = f"output_{model}_{args.prompt_type}_{args.seed}.txt"
        
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
incorrect_outputs = []
num_incorrect = 1
for line in data:
    cleaned_inputs = clean(line['input'])
    cleaned_outputs = clean(line['answer'])

    if line['is_correct'] == False:
        num_incorrect += 1
        incorrect_outputs.append({'tokenized_input': cleaned_inputs, 'tokenized_output': cleaned_outputs})
    
    inputs.append(cleaned_inputs)
    outputs.append(cleaned_outputs)
print(f"Total incorrect: {num_incorrect} out of {len(data)}.")

accuracy_first, incorrect_firsts = get_accuracy_first(inputs, outputs)
accuracy_last, incorrect_lasts = get_accuracy_last(inputs, outputs)
accuracy_ind, absent_copies, total_unique = get_accuracy_unique_right(inputs, outputs)
accuracy_unique_bigrams, absent_copies_bg, total_unique_bg = get_accuracy_unique_bigrams(inputs, outputs)
accuracy_repeated_bigrams, absent_copies_rep, total_rep = get_accuracy_repeated_bigrams(inputs, outputs)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    print(f'First accuracy: {accuracy_first*100}%')
    print(f'Last accuracy: {accuracy_last*100}%')
    print(f'Unique token + right accuracy: {accuracy_ind*100}% for total of {total_unique} tokens.')
    print(f'Unique bigrams accuracy:  {accuracy_unique_bigrams*100}% for total of {total_unique_bg} bigrams.')
    print(f'Repeated bigrams accuracy: {accuracy_repeated_bigrams*100}% for total of {total_rep} bigrams.')
    
    if len(incorrect_outputs) > 0:
        f.write(f'Marked as incorrect initially\n')
        for inp in incorrect_outputs:
            f.write(str(inp['tokenized_input']) + "\n" + str(inp['tokenized_output']) + "\n" + "=====================\n")
    
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
    
    f.write(f'Unique token + right accuracy: {accuracy_ind*100}% for total of {total_unique} tokens.\n')
    if len(absent_copies) > 0:
        f.write(f'Absent copies: {len(absent_copies)}\n')
        for inp in absent_copies:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")

    f.write(f'Unique bigram accuracy: {accuracy_ind * 100}% for total of {total_unique} tokens.\n')
    if len(absent_copies_bg) > 0:
        f.write(f'Absent copies: {len(absent_copies_bg)}\n')
        for inp in absent_copies_bg:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")
            
    f.write(f'Repeated bigram accuracy: {accuracy_ind * 100}% for total of {total_unique} tokens.\n')
    if len(absent_copies_rep) > 0:
        f.write(f'Absent copies: {len(absent_copies_rep)}\n')
        for inp in absent_copies_rep:
            f.write(str(inp['token']) + "\n" + str(inp['input']) + "\n" + str(
                inp['output']) + "\n" + "=====================\n")
            
print(f"Output written to {OUTPUT_FILE}")