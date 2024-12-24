import os
from huggingface_hub import HfApi, logging

MODEL_NAME="allenai/OLMo-7B-0724-Instruct-hf"

base_path = "/scratch/common_models/"
os.environ["HF_HOME"] = base_path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.set_verbosity_debug()
hf = HfApi()
hf.snapshot_download(MODEL_NAME, cache_dir="/scratch/common_models/")

os.execv("/bin/bash",
         ["bash", "-c",
          f"cp -L {base_path}/modelsnapshots/aba3d33d766a33a44677e3a163f0fe2d1010d90a/* /scratch/common_models/hf_model/ "
          "&& rm -rf * && mv /scratch/common_models/hf_model/ /scratch/common_models/OLMo-7B-0724-Instruct-hf"])
