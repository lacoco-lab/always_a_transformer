from utils import get_data


def clean_results(results):
    """
    Parse the answer from the full_answer field in the results
    :param results: arr
    :return: arr 
    """
    for line in results:
        line['answer'] = line['full_answer'][0]   
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


llama_after_anti = clean_results(get_data("results/llama_non-instruct_after_anti-induction.jsonl"))
llama_before_anti = clean_results(get_data("results/llama_non-instruct_before_anti-induction.jsonl"))
llama_after = clean_results(get_data("results/llama_non-instruct_after_induction.jsonl"))
llama_before = clean_results(get_data("results/llama_non-instruct_before_induction.jsonl"))

print(f"Accuracy Llama Induction-After pruned Anti-Induction: {get_accuracy(llama_after_anti)}")
print(f"Accuracy Llama Induction-Before pruned Anti-Induction: {get_accuracy(llama_before_anti)}")
print(f"Accuracy Llama Induction-After pruned Induction: {get_accuracy(llama_after)}")
print(f"Accuracy Llama Induction-Before pruned Induction: {get_accuracy(llama_before)}")