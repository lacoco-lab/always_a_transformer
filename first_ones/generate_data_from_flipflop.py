import argparse
import os

from pathlib import Path

import jsonlines

from tqdm.auto import tqdm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=str, required=True, help="Path to the source directory")
    ap.add_argument("--dest_dir", type=str, required=True, help="Path to the destination directory")
    args = ap.parse_args()
    
    all_files = Path(args.src_dir).rglob("*.txt")
    only_500_digit_files = [ file for file in all_files if "500_w0" in file.name or "500_w496" in file.name or "500_pw0" in file.name or "500_pw496" in file.name ]
    only_non_worded_files = [ file for file in only_500_digit_files if "worded" not in str(file) ]
    
    os.makedirs(args.dest_dir, exist_ok=True)
    
    jsonl_data = []
    for file in tqdm(only_non_worded_files):
        with open(file, "r") as f:
            data = [line.strip() for line in f.readlines()]
            filenames = [str(file) for _ in range(len(data))]
            jsonl_data.extend([{"input": d, "filename": f} for d, f in zip(data, filenames)])
    
    with jsonlines.open(Path(args.dest_dir) / "data.jsonl", "w") as writer:
        writer.write_all(jsonl_data)