import argparse
import os

from pathlib import Path

from tqdm.auto import tqdm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_dir", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--op_dir_postfix", type=str, required=True, help="Postfix for the output dir")

    args = ap.parse_args()

    ip_files = list(Path.rglob(Path(args.ip_dir), "*.txt"))
    print(f"Total files: {len(ip_files)}")

    for file in tqdm(ip_files):
        parent_dir = Path(Path(args.ip_dir).parent, Path(args.ip_dir).name+"-"+args.op_dir_postfix)
        op_f_path = Path(parent_dir, str(file.parent).split("/")[-1], file.name)
        os.makedirs(op_f_path.parent, exist_ok=True)
        with open(file) as reader:
            data = reader.readlines()
        with open(op_f_path, "w") as writer:
            worded_data = []
            for d in data:
                # DO NOT CHANGE THE ORDER OF REPLACE
                worded_data.append(d.replace("r", "read").replace("i", "ignore").replace("w", "write"))
            writer.writelines(worded_data)