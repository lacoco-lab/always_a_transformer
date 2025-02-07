# Example usage: python utils/fast_download_model.py --model_name "microsoft/phi-4" --base_path /scratch/common_models
# For adding the model to local storage, use the flag --add_to_local true
import argparse
import os

import shutil
from datetime import datetime
from glob import glob

from huggingface_hub import HfApi, logging


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, required=True, help="Model name to download")
    args.add_argument("--base_path", type=str, required=True, help="Path to the base directory")
    args.add_argument("--add_to_local", type=bool, required=False, default=False)
    
    args = args.parse_args()
    
    MODEL_NAME = args.model_name
    base_path = args.base_path
    add_to_local = args.add_to_local
    
    os.environ["HF_HOME"] = base_path
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    logging.set_verbosity_debug()
    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    hf = HfApi()
    hf.snapshot_download(MODEL_NAME, cache_dir=base_path)
    
    os.makedirs(f"{base_path}/hf_model_{curr_time}", exist_ok=True)
    
    og_path = glob(f"{base_path}/models--*/snapshots/*/original")
    
    if og_path and os.path.isdir(og_path[0]):
        print("Deleting original checkpoints:", og_path)
        shutil.rmtree(og_path[0])
    
    if add_to_local:
        os.execv("/bin/bash",
                 ["bash", "-c",
                  f"cp -rL {base_path}/models--*/snapshots/*/* {base_path}/hf_model_{curr_time}/ && rm -rf {base_path}/models--* && mv {base_path}/hf_model_{curr_time} {base_path}/{MODEL_NAME.split('/')[1]} && echo 'add' > {base_path}/{MODEL_NAME.split('/')[1]}/storage.txt"])
    else:
        os.execv("/bin/bash",
                 ["bash", "-c",
                  f"cp -rL {base_path}/models--*/snapshots/*/* {base_path}/hf_model_{curr_time}/ && rm -rf {base_path}/models--* && mv {base_path}/hf_model_{curr_time} {base_path}/{MODEL_NAME.split('/')[1]}"])
