import argparse
import os

from pathlib import Path

import jsonlines

from tqdm.auto import tqdm


def get_flipflop_files(all_files):
    only_500_digit_files = [file for file in all_files if "500.txt" in file.name]
    only_non_worded_files = [file for file in only_500_digit_files if "worded" not in str(file)]
    return only_non_worded_files

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, choices=['first', 'last', 'flipflop', 'flipflop_inductionhead'], 
                   required=True, help="Task to pick data for")
    p.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    p.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    # p.add_argument("--num_samples", type=int, required=True, help="Number of samples to pick")
    p.add_argument("--length", type=int, default=500, required=False, help="Length of the input data")
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    all_files = Path(args.input_dir).rglob("*.txt")
    
    if args.task == "flipflop":
        only_non_worded_files = get_flipflop_files(all_files)
    elif args.task == "first":
        # TBD
        only_non_worded_files = []
    elif args.task == "last":
        # TBD
        only_non_worded_files = []
    elif args.task == "flipflop_inductionhead":
        # TBD
        only_non_worded_files = []
    else:
        raise ValueError("Invalid task")
    
    print(only_non_worded_files)
    
    jsonl_data = []
    for file in tqdm(only_non_worded_files):
        with open(file, "r") as f:
            data = [line.strip() for line in f.readlines()]
            filenames = [str(file) for _ in range(len(data))]
            jsonl_data.extend([{"input": d, "filename": f} for d, f in zip(data, filenames)])
    
    print(f"Number of samples: {len(jsonl_data)} for task: {args.task} and length: {args.length}")
    with jsonlines.open(Path(args.output_dir) / "data.jsonl", "w") as writer:
        writer.write_all(jsonl_data)
    
    