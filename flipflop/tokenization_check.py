import argparse

from tqdm.auto import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name")
    args = ap.parse_args()

    with open(args.ip_path) as reader:
        if args.ip_path.endswith(".txt"):
            data = reader.readlines()
        else:
            raise ValueError("Invalid input file type")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    for idx, d in enumerate(tqdm(data)):
        tokens = tokenizer.tokenize(d.strip(), add_special_tokens=False)
        if len(tokens) != len(d.strip()):
            print(f"Index: {idx}, Input: {d.strip()}, Tokens: {tokens}")
            break
    else:
        print("All inputs tokenized correctly")
