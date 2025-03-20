import nltk
import re
from collections import defaultdict, Counter
import json


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
            ans = ans.split()
            gold_ans = gold_ans.split()
            if ans[0] == gold_ans[0]:
                correct += 1
            else:
                if ans[0] == '':  # olmo tokenization issue
                    if ans[1] == gold_ans[0]:
                        correct += 1
                        continue
                incorrect_firsts.append({'input': gold_ans, 'output': ans, 'token': gold_ans[0]})
        except IndexError:
            continue
            # print(ans)
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
    for gold_ans, ans in zip(inputs, outputs):
        try:
            ans = ans.split()
            gold_ans = gold_ans.split()
            if ans[len(ans) - 1] == gold_ans[len(gold_ans) - 1]:
                correct += 1
            else:
                incorrect_lasts.append({'input': gold_ans, 'output': ans, 'token': gold_ans[len(gold_ans) - 1]})
        except IndexError:
            continue
            # print(ans)
    return correct / len(inputs), incorrect_lasts


def get_accuracy_unique_right(inputs, outputs):
    """
    Calculate whether a unique token with its right token was copied correct

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
                # print(f"Failed to copy the unique token {tok}")
                absent_copies.append({'input': gold_ans, 'output': ans, 'token': tok})
                continue

            if gold_ans[unique_pos_gold:unique_pos_gold + 2] == ans[unique_pos_ans:unique_pos_ans + 2]:
                correct_after += 1

            total_unique += 1

    return correct_after / total_unique, absent_copies, total_unique


def clean_text(sentence):
    """
    Clean text: remove newlines, punctuation, etc.
    :param sentence: str
    :return: str
    """
    return re.sub(r'[^\w\s]', '', sentence).replace('\n', ' ').replace('\r', '')


def get_bigram_counts(bigrams):
    """
    Get counts of each bigram in the sentence.
    :param bigrams: arr 
    :return: dict
    """
    bigram_counts = Counter(bigrams)

    return bigram_counts


def classify_bigrams(text: str):
    """
    Categorize bigrams into deterministic and non-deterministic based on the second token in the bigram.

    :param text: str, word-level input
    :return: arr, arr, arr
    """
    bigrams = list(nltk.bigrams(text.strip().replace(".", "").split()))

    first_dict = defaultdict(set)
    for first, second in bigrams:
        first_dict[first].add(second)
    deterministic = []
    non_deterministic = []

    for (first, second) in bigrams:
        if len(first_dict[first]) == 1:
            deterministic.append((first, second))
        else:
            non_deterministic.append((first, second))

    return bigrams, deterministic, non_deterministic


def pad_ans(gold, ans):
    """
    Pad the copied sequence if it is shorter than the original sequence.
    :param gold: str
    :param ans: str
    :return: str
    """
    if len(ans.split()) < len(gold.split()):
        while len(ans.split()) != len(gold.split()):
            ans += " <pad>"
    return ans


def get_bigram_accuracy(inputs, ouputs):
    """
    Calculate copy accuracy for unique and non-unique bigrams while skipping hallucinated bigrams.
    :param original_bigrams: arr
    :param copied_bigrams: arr
    :return: float, copy accuracy of the unique and non-unique bigrams"""
    
    unique_accs = []
    non_unique_accs = []
    
    error_cases = []
    
    for inp, out in zip(inputs, ouputs):
        original_bigrams = list(nltk.bigrams(inp.split()))
        ans = pad_ans(inp, out)
        copied_bigrams = list(nltk.bigrams(ans.split()))
        unique_bigrams, non_unique_bigrams = classify_bigrams(original_bigrams)
    
        correct_unique = 0
        correct_non_unique = 0
    
        total_unique = sum(1 for bigram in original_bigrams if bigram in unique_bigrams)
        total_non_unique = sum(1 for bigram in original_bigrams if bigram in non_unique_bigrams)
    
        orig_idx, copied_idx = 0, 0  # Two pointers for original and copied bigrams
    
        while orig_idx < len(original_bigrams) and copied_idx < len(copied_bigrams):
            orig_bigram = original_bigrams[orig_idx]
            copied_bigram = copied_bigrams[copied_idx]
    
            if copied_bigram == orig_bigram:  # Correct match
                if orig_bigram in unique_bigrams:
                    correct_unique += 1
                elif orig_bigram in non_unique_bigrams:
                    correct_non_unique += 1
                orig_idx += 1  # Move both pointers forward
                copied_idx += 1
            elif copied_bigram not in original_bigrams:  # Hallucinated bigram
                copied_idx += 1  # Skip this bigram in copied sequence
            else:
                # this case might indicate a hallucinated token, need to think about it
                orig_idx += 1  # Move forward in original sequence
    
        unique_accuracy = correct_unique / total_unique
        non_unique_accuracy = correct_non_unique / total_non_unique
        
        if unique_accuracy != 1:
            error_cases.append(
                {
                    'original': original_bigrams,
                    'copied': copied_bigrams,
                    'unique_bigrams': unique_bigrams,
                }
            )
            
        unique_accs.append(unique_accuracy)
        non_unique_accs.append(non_unique_accuracy)
    
    return sum(unique_accs)/len(inputs), sum(non_unique_accs)/len(inputs), error_cases


def get_model_name(model_key, version):
    """
    Retrieves the correct model name based on user input.
    :param model_key: str
    :param version: str
    :return: str
    """
    if model_key == "llama_8b":
        if version == 'instruct':
            return "llama3.1_8B-instruct"
        elif version == 'non-instruct':
            return "llama3.1_8B"
    elif model_key == "llama_70b":
        if version == "instruct":
            return "llama3.3_70B-instruct"
        elif version == "non-instruct":
            return "llama3.1_70B"
    elif model_key == "olmo":
        return "OLMo_7B-instruct"
    elif model_key == 'pythia':
        return "pythia-1.4b"
    raise ValueError(f"Model '{model_key}' not recognized or missing version")

