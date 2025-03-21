import re
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

def extract_answer(ans):
    """
    Extract answer from the full model answer.
    :param ans: dict, model answer
    :return: dict
    """

    match = re.search(r"<ans>(.*?)</ans>", ans['full_answer'])
    if match:
        extracted = match.group(1)
        ans['answer'] = extracted
    else:
        ans['answer'] = None
    return ans


def count_distribution(answers):
    """
    Count the number of distinct answers: 0 or 1.
    :param answers: arr
    :return: float, float - frequencies of 0 or 1 being the answer
    """

    count_zero = 0
    count_one = 0
    for ans in answers:
        if str(ans['gold_ans_char']) == '0':
            count_zero += 1
        elif str(ans['gold_ans_char']) == '1':
            count_one += 1
    return count_zero / len(answers), count_one / len(answers)


def clean_results(results):
    """
    Parse the answer from the full_answer field in the results
    :param results: arr
    :return: arr 
    """
    for line in results:
        line['answer'] = line['answer'][0]
    return results


def get_accuracy(results):
    """
    Calculate the accuracy of the answer.
    :param results: arr
    :return: float
    """
    correct = 0
    for line in results:
        if str(line['answer']) == str(line['gold_ans_char']):
            correct += 1
    return (correct / len(results)) * 100


llama_after = clean_results(get_data("flipflop_inductionhead/Meta-Llama-3-8B/inductionhead_zero-shot_completion_v0/100_after_seed-5.jsonl"))
llama_before = clean_results(get_data("flipflop_inductionhead/Meta-Llama-3-8B/inductionhead_zero-shot_completion_v0/100_before_seed-5.jsonl"))
llama_before_instruct = get_data('flipflop_inductionhead/Meta-Llama-3-8B-Instruct/inductionhead_zero-shot_chat_v0/100_before_nocot_seed-5.jsonl')
llama_after_instruct = get_data('flipflop_inductionhead/Meta-Llama-3-8B-Instruct/inductionhead_zero-shot_chat_v0/100_after_nocot_seed-5.jsonl')


print(f'Accuracy Non-Instruct Llama After: {get_accuracy(llama_after)}')
print(f'Accuracy Non-Instruct Llama Before: {get_accuracy(llama_before)}')
print(f'Accuracy Instruct Llama After: {get_accuracy(llama_after_instruct)}')
print(f'Accuracy Instruct Llama Before: {get_accuracy(llama_before_instruct)}')