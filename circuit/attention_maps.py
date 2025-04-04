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

model = HookedTransformer.from_pretrained(model_name, local_files_only=True)
model.cfg.use_cache = False
tokenizer = model.tokenizer

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
print(f"Random seed: {args.seed}, top_k: {args.top_k}")

output_dir = 'attention_plots'
attention_scores = {}

for example in data:
    if args.version == 'non-instruct':
        max_new = 2
    elif args.version == 'instruct':
        max_new = 5000

    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)
    w_token_id = tokenizer.encode('w', add_special_tokens=False)[0]
    
    try:
        w_indices = [i for i, x in enumerate(tokens[0]) if x == w_token_id]
    except ValueError:
        print(f"Token 'w' not found in the input: {tokens[0]}")
        continue

    _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "attn" in name,
            return_type=None
        )

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attn_patterns = cache[get_act_name("attn", layer)]
            head_patterns = attn_patterns[:, head, :, :]
            if (layer, head) in attention_scores.keys():
                attention_scores[(layer, head)] += head_patterns[0, w_indices[1], w_indices[0]+1].item()
            else:
                attention_scores[(layer, head)] = head_patterns[0, w_indices[1], w_indices[0]+1].item()

sorted_attention_heads = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)[:13]
for (layer, head), score in sorted_attention_heads:
    print(f"Attention for Layer {layer}, Head {head} with score {score:.4f}")


# Use the first example to plot attention maps for the top-N heads
example = data[0]
system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
prompt = template.render(system=system_prompt, user_input=task_prompt)
tokens = model.to_tokens(prompt)

_, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "attn" in name,
    return_type=None
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for (layer, head), score in sorted_attention_heads:
    attn_matrix = cache[get_act_name("attn", layer)][0, head, :, :].detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap="viridis")
    plt.colorbar()
    plt.title(f"Attention Map - Layer {layer}, Head {head} (Score: {score:.4f})")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.split("/")[-1]}_attention_layer{layer}_head{head}.png"))
    plt.close()
