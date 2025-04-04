import random
import numpy as np
import torch
from jinja2 import Template
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from argparse import ArgumentParser
import jsonlines
from utils import combine_params, get_data, load_heads, render_prompt, get_gold_ans
import matplotlib.pyplot as plt
import os

def set_global_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collect_head_hook(layer: int, head_index: int, collected):
    """Hook function to collect activations for a specific head."""
    def hook(value, hook):
        collected[(layer, head_index)].append(value[:, :, head_index, :].detach().cpu())
        return value
    return hook

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", help="choose model")
parser.add_argument("-v", "--version", dest="version", help="instruct or non-instruct model version")
parser.add_argument("-t", "--task", dest="task", help="induction before or after")
parser.add_argument("-tp", "--type", dest="type", help="ablate induction or anti-induction")
parser.add_argument("-l", "--length", dest="length", help="choose input length: 20, 30, 50, 100")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()
model_name, task_path, version, data_path, ablation_type = combine_params(args)

set_global_seed(args.seed)

model = HookedTransformer.from_pretrained(model_name, local_files_only=True)
model.cfg.use_cache = False

data = get_data(data_path)[:100]
inp_length = len(data[0]['input'])

template_str = "System: {{ system }} User: {{ user_input }}"
system_path = 'templates/system.jinja'
template = Template(template_str)
heads_to_save = load_heads(model_name, ablation_type)

print(f"Loaded model: {model_name}")
print(f"Loaded input data from: {data_path}")
print(f"Chosen task: {task_path}")
print(f"Chosen ablation type: {ablation_type}")
print(f"Heads to save: {heads_to_save}")
print(f"Random seed: {args.seed}, top_k: {args.top_k}")

# --- Run the dataset and collect activations for each head ---
collected_acts = {(layer, head): [] for layer, head in heads_to_save}

print("Collecting original activations over the dataset...")
for example in data:
    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)
    
    hooks = [
        (f'blocks.{layer}.attn.hook_z', collect_head_hook(layer, head, collected_acts))
        for layer, head in heads_to_save
    ]
    
    _ = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=hooks)

# --- Save the collected original activations ---
output_dir = "activation_cache"
os.makedirs(output_dir, exist_ok=True)

for (layer, head), act_list in collected_acts.items():
    all_acts = torch.cat([act.view(-1, act.size(-1)) for act in act_list], dim=0)
    file_path = os.path.join(output_dir, f"layer_{layer}_head_{head}.pt")
    torch.save(all_acts, file_path)
    print(f"Saved activations for Layer {layer}, Head {head} to {file_path}")
