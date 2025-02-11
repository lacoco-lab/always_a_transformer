import argparse
import jsonlines
from pathlib import Path

from tqdm.auto import tqdm


def read_jsonl_file(jsonl_file_path):
    results = []
    with jsonlines.open(jsonl_file_path) as reader:
        for example in reader:
            results.append(example)
    return results


if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    p.add_argument("--source_file", type=str, required=True, help="Path to the source file")
    
    args = p.parse_args()
    
    source_file = Path(args.source_file)
    
    source_data = read_jsonl_file(source_file)
    correct_cnt = 0
    no_end_cnt = 0
    wrong_cnt = 0
    
    for entry in tqdm(source_data):
        if entry["answer"] == entry["gold_ans_char"]:
            correct_cnt += 1
        else:
            wrong_cnt += 1
        if "</ans>" not in entry["full_answer"]:
            no_end_cnt += 1
    
    print(f"Correct count: {correct_cnt}")
    print(f"Wrong count: {wrong_cnt}")
    print(f"No end count: {no_end_cnt}")
    print(f"Total count: {len(source_data)}")