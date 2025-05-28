import torch
import random
import numpy as np
import os

def get_logging_function(output_dir):
    log_path = output_dir / f"logs.txt"
    f = open(log_path, "w")

    def print_to_both(*args):
        print(*args)
        print(*args, file=f, flush=True)

    def cleanup():
        f.close()

    print_to_both.cleanup = cleanup
    return print_to_both

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)