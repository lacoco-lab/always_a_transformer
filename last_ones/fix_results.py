import argparse

from pathlib import Path
import jsonlines

def read_jsonl_file(jsonl_file_path):
    data = []
    with jsonlines.open(jsonl_file_path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_file", type=str, required=True, help="Path to the jsonl file")
    args = ap.parse_args()

    jsonl_file_path = Path(args.jsonl_file)
    data = read_jsonl_file(jsonl_file_path)

    wrong_cnt = 0

    for obj in data:
        ip = obj["input"]
        obj["gold_ans_char"] = ip[-1]
    
    with jsonlines.open(jsonl_file_path, "w") as writer:
        writer.write_all(data)