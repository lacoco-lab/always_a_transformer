from utils import get_data
import re


def extract_answer(ans):
    """
    Extract answer from the full model answer.
    :param ans: dict, model answer
    :return: dict
    """

    match = re.search(r"<ans>(.*?)</ans>", ans['full_answer'])
    if match:
        extracted = match.group(1)
        return extracted
    else:
        print(f'No answer tag found.')
        return None


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
    return count_zero/len(answers), count_one/len(answers)


def clean_results(results):
    """
    Parse the answer from the full_answer field in the results
    :param results: arr
    :return: arr 
    """
    for line in results:
        #line['answer'] = line['full_answer'][0]
        line['answer'] = extract_answer(line)
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


#llama_after_anti = clean_results(get_data("results/llama_non-instruct_after_random-all_100.jsonl"))
#llama_before_anti = clean_results(get_data("results/llama_non-instruct_before_anti-induction.jsonl"))
llama_after = clean_results(get_data("results/gemma_instruct_after_induction_20.jsonl"))
#llama_before = clean_results(get_data("results/llama_non-instruct_before_induction.jsonl"))

#llama_instruct_after_anti = [extract_answer(ans) for ans in clean_results(get_data("results/llama_instruct_after_anti-induction.jsonl"))]
#llama_instruct_before_anti = [extract_answer(ans) for ans in clean_results(get_data("results/llama_instruct_before_anti-induction.jsonl"))]
#llama_instruct_after = [extract_answer(ans) for ans in clean_results(get_data("results/llama_instruct_after_induction.jsonl"))]
#llama_instruct_before = [extract_answer(ans) for ans in clean_results(get_data("results/llama_instruct_before_induction.jsonl"))]

prop_zero, prop_one = count_distribution(llama_after)
print(f"Proportions in data:\n0: {prop_zero} for {len(llama_after)} samples.\n1: {prop_one} for {len(llama_after)} samples.\n=========")

#print(f"Accuracy Non-Instruct Llama Induction-After pruned Anti-Induction: {get_accuracy(llama_after_anti)}")
print(f"Accuracy Induction-After: {get_accuracy(llama_after)}")
#print(f"Accuracy Non-Instruct Llama Induction-Before pruned Anti-Induction: {get_accuracy(llama_before_anti)}")
#print(f"Accuracy Non-Instruct Llama Induction-Before pruned Induction: {get_accuracy(llama_before)}")
print("========")
#print(f"Accuracy Instruct Llama Induction-After pruned Anti-Induction: {get_accuracy(llama_instruct_after_anti)}")
#print(f"Accuracy Instruct Llama Induction-After pruned Induction: {get_accuracy(llama_instruct_after)}")
#print(f"Accuracy Instruct Llama Induction-Before pruned Anti-Induction: {get_accuracy(llama_instruct_before_anti)}")
#print(f"Accuracy Instruct Llama Induction-Before pruned Induction: {get_accuracy(llama_instruct_before)}")