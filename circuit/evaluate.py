import re
import os
import json


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
        #print(f'No answer tag found.')
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

def load_files(folder_path):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            obj = {'filename': filename, 'data': data}
            all_data.append(obj)

    return all_data

folder_path = 'results'
data = load_files(folder_path)
print(f"Loaded {len(data)} records from all jsonl files.")

for record in data:
    if 'gemma' in record['filename']:
        cleaned_results = clean_results(record['data'])
        distrs = count_distribution(cleaned_results)
        print(f'Accuracy for {record['filename']} is {get_accuracy(cleaned_results)}%')
        #print(f'Count distrubution for file {record['filename']} is:\n0 - {distrs[0]}%, 1 - {distrs[1]}%')
        print(f'============================')
