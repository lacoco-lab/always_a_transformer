import argparse

from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name")
    ap.add_argument("--with_space", type=bool, default=False, help="Check for space in tokens")
    args = ap.parse_args()

    print(f"Checking for incorrectly tokenized data in {args.ip_path} using tokenizer: {args.tokenizer_name} "
          f"with space: {args.with_space}")

    all_files = list(Path(args.ip_path).rglob("*.txt"))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    ic_cnt = 0

    for file in tqdm(all_files):
        with open(file) as reader:
            data = reader.readlines()

        for idx, d in enumerate(tqdm(data)):
            if args.with_space:
                tokens = tokenizer.tokenize(" ".join(list(d.strip())), add_special_tokens=False)
            else:
                tokens = tokenizer.tokenize(d.strip(), add_special_tokens=False)
            ic_flag = False
            for token in tokens:
                if (args.with_space and len(token) > 2) or (not args.with_space and len(token) > 1):
                    print(f"Index: {idx}, Input: {d.strip()}, Tokens: {tokens}")
                    print(f"Token: {token}")
                    ic_flag = True
                    break
            if ic_flag:
                ic_cnt += 1
                break

    print(f"Total incorrectly tokenized files: {ic_cnt}")