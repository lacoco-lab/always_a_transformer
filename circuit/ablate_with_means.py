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

"""
It would be nice if I cached activations just once, saved them in the file and reused.
This is a bit ugly code for now.
"""

def set_global_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collect_head_hook(layer: int, head_index: int, collected):
    def hook(value, hook):
        collected[(layer, head_index)].append(value[:, :, head_index, :].detach().cpu())
        return value
    return hook

def patch_head_hook(layer: int, head_index: int, global_mean):
    def hook(value, hook):
        value[:, :, head_index, :] = global_mean.to(value.device)
        return value
    return hook


parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", help="choose model")
parser.add_argument("-v", "--version", dest="version", help="instruct or non-instruct model version")
parser.add_argument("-t", "--task", dest="task", help="induction before or after")
parser.add_argument("-tp", "--type", dest="type", help="ablate induction or anti-induction")
parser.add_argument("-l", "--length", dest="length", help="choose input length: 20, 30, 50, 100")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--top_k", type=int, default=5, help="Top-k sampling parameter")

args = parser.parse_args()
model_name, task_path, version, data_path, ablation_type = combine_params(args)

set_global_seed(args.seed)

#local_files_only=True
model = HookedTransformer.from_pretrained(model_name)
model.cfg.use_cache = False

data = get_data(data_path)[:100]
inp_length = len(data[0]['input'])

template_str = "System: {{ system }} User: {{ user_input }}"
system_path = 'templates/system.jinja'
template = Template(template_str)
heads_to_ablate = load_heads(model_name, ablation_type)

print(f"Loaded model: {model_name}")
print(f"Loaded input data from: {data_path}")
print(f"Chosen task: {task_path}")
print(f"Chosen ablation type: {ablation_type}")
print(f"Heads to ablate: {heads_to_ablate}")
print(f"Random seed: {args.seed}, top_k: {args.top_k}")

# --- First Pass: Run the dataset and collect activations for each head ---
collected_acts = {(layer, head): [] for layer, head in heads_to_ablate}

print("Collecting activations over the dataset...")
for example in data:
    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)
    
    hooks = [
        (f'blocks.{layer}.attn.hook_z', collect_head_hook(layer, head, collected_acts))
        for layer, head in heads_to_ablate
    ]
    
    _ = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=hooks)

global_means = {}
for (layer, head), act_list in collected_acts.items():
    # Each collected tensor has shape [batch, tokens, d_head].
    # Flatten each tensor across batch and token dimensions then concatenate
    flat_acts = torch.cat([act.view(-1, act.size(-1)) for act in act_list], dim=0)
    global_means[(layer, head)] = flat_acts.mean(dim=0)
    print(f"Global mean computed for Layer {layer}, Head {head}: {global_means[(layer, head)]}")

# Second pass - replace head activation with the mean over all the examples
answers = []
for example in data:
    if args.version == 'non-instruct':
        max_new = 2
    elif args.version == 'instruct':
        max_new = 1000

    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)
    
    # Set up hooks for each head using the precomputed global mean
    hooks = [
        (f'blocks.{layer}.attn.hook_z', patch_head_hook(layer, head, global_means[(layer, head)]))
        for layer, head in heads_to_ablate
    ]
    
    original_loss = model(tokens, return_type="loss")
    ablated_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=hooks)
    print(f"Original Loss: {original_loss.item():.3f}")
    print(f"Ablated Loss: {ablated_loss.item():.3f}")

    with model.hooks(fwd_hooks=hooks):
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_new,
            stop_at_eos=True,
            do_sample=False,
            top_k=args.top_k,
            temperature=0, 
        )
    new_tokens = generated_tokens[0, tokens.shape[-1]:]
    generated_text = model.to_string(new_tokens)

    answer = {
        'input': example['input'],
        'gold_ans_char': get_gold_ans(example['input'], args.task),
        'full_answer': generated_text
    }
    answers.append(answer)

output_path = (
    'results/'
    + args.model + '_'
    + args.version + '_'
    + args.task + '_'
    + args.type + "_"
    + str(inp_length) + "_"
    + "with_mean"
    + '.jsonl'
)
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(answers)
