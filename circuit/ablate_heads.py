import random
import numpy as np
import torch
from jinja2 import Template
from transformer_lens import HookedTransformer
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

def ablate_head_hook(layer: int, head_index: int):
    def hook(value, hook):
        before_mean = value[:, :, head_index, :].mean().item()
        value[:, :, head_index, :] = 0.
        after_mean = value[:, :, head_index, :].mean().item()
        print(f"Layer {layer}, Head {head_index}: Mean before ablation = {before_mean:.4f}, after = {after_mean:.4f}. Tensore shape: {value.shape}")
        return value
    return hook

def check_logits_for_next_token(model: HookedTransformer, tokens, topk: int = 5):
    """
    Prints the top-k next-token candidates by logit score, given the current tokens.
    """
    logits = model(tokens, return_type='logits')

    next_token_logits = logits[0, -1, :]  # shape: [vocab_size]

    topk_values, topk_indices = next_token_logits.topk(topk)

    print("\nTop next-token candidates (logits):")
    for val, idx in zip(topk_values, topk_indices):
        token_str = model.to_string(idx.unsqueeze(0))
        print(f"  Token: {repr(token_str)} | Logit: {float(val):.4f}")
    print()

def plot_attention_map(attention_data, tokens, output_dir, layer, head):
    """Plot and save attention map for the specified layer and head."""
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_data, cmap='viridis', aspect='auto')
    plt.colorbar()
    token_labels = [model.to_string([t]) for t in tokens]
    plt.xticks(ticks=np.arange(len(token_labels)), labels=token_labels, rotation=90)
    plt.yticks(ticks=np.arange(len(token_labels)), labels=token_labels)
    plt.title(f"Attention from token 'w' to next token\nLayer {layer}, Head {head}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    attention_plot_path = os.path.join(output_dir, f"attention_layer{layer}_head{head}.png")
    plt.tight_layout()
    plt.savefig(attention_plot_path)
    plt.close()

def capture_attention_hook(layer: int, head: int):
    """Hook to capture attention data for a specific head."""
    def hook(value, hook):
        # Capture attention data from hook_values (shape: [batch_size, num_heads, seq_len, seq_len])
        attention_data = value # Attention for all heads
        return attention_data
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

# Set the global seed
set_global_seed(args.seed)

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

answers = []
output_dir = 'attention_plots'

for example in data:
    if args.version == 'non-instruct':
        max_new = 2
    elif args.version == 'instruct':
        max_new = 5000

    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)
    token_strs = model.to_string(tokens[0])
    
    try:
        w_index = token_strs.index('w')
        next_token_index = w_index + 1
    except ValueError:
        print(f"Token 'w' not found in the input: {token_strs}")
        continue
    
    print(f"Found 'w' at index {w_index}, next token at index {next_token_index}")

    hooks = [
        (f'blocks.{layer}.attn.hook_z', ablate_head_hook(layer, head))
        for layer, head in heads_to_ablate
    ]
    capture_hooks = [
        (f'blocks.{layer}.attn.hook_pattern', capture_attention_hook(layer, head))
        for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)
    ]
    
    original_loss = model(tokens, return_type="loss")
    ablated_loss = model.run_with_hooks(
        tokens, 
        return_type="loss", 
        fwd_hooks=hooks
    )

    print(f"Original Loss: {original_loss.item():.3f}")
    print(f"Ablated Loss: {ablated_loss.item():.3f}")

    attention_scores = {}
    with model.hooks(fwd_hooks=hooks+capture_hooks):
        generated_tokens = model.generate(
                tokens,
                max_new_tokens=max_new,
                stop_at_eos=True,
                do_sample=False,
                top_k=args.top_k,
                temperature=0, 
            )

        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                # Access the captured attention data from the hook
                attention_data = model.hooks[f'blocks.{layer}.attn.hook_pattern'][0, :, head, :].detach().cpu().numpy()
                attention_score = attention_data[w_index, next_token_index]  # Focus on attention from 'w' to the next token
                attention_scores[(layer, head)] = attention_score
  
        sorted_attention_heads = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for (layer, head), score in sorted_attention_heads:
            print(f"Saving attention map for Layer {layer}, Head {head} with score {score:.4f}")
            attention_data = model.attn_data[layer, head, w_index, next_token_index]
            plot_attention_map(attention_data, tokens, output_dir, layer, head)

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
    + str(inp_length)
    + '.jsonl'
)
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(answers)
