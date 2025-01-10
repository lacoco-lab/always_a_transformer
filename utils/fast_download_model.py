import shutil
import os
from glob import glob

from huggingface_hub import HfApi, logging

MODEL_NAME = "meta-llama/Llama-3.1-70B"

base_path = "/scratch/common_models/"
os.environ["HF_HOME"] = base_path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.set_verbosity_debug()
hf = HfApi()
hf.snapshot_download(MODEL_NAME, cache_dir="/scratch/common_models/")

os.makedirs(f"{base_path}/hf_model", exist_ok=True)

og_path = glob(f"{base_path}/models--*/snapshots/*/original")

if os.path.isdir(og_path[0]):
    print("Deleting original checkpoints:", og_path)
    shutil.rmtree(og_path[0])

os.execv("/bin/bash",
         ["bash", "-c",
          f"cp -L {base_path}/models--*/snapshots/*/* /scratch/common_models/hf_model/ && rm -rf {base_path}/models--* && mv /scratch/common_models/hf_model /scratch/common_models/{MODEL_NAME.split('/')[1]}"])
