import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                        help="choose model")
args = parser.parse_args()

if args.model == 'llama3.1-8b-instruct':
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
else:
    model = "meta-llama/Meta-Llama-3.1-8B"

model = HookedTransformer.from_pretrained(model)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_synthetic_sequence(vocab_size=model.tokenizer.vocab_size, seq_len=50):
    random_tokens = torch.randint(0, vocab_size, (seq_len,))
    return torch.cat([random_tokens, random_tokens])

def calculate_induction_score(head, layer, num_samples=1000):
    total_score = 0
    seq_len = 50

    for _ in range(num_samples):
        tokens = generate_synthetic_sequence()

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "attn" in name,
            return_type=None
        )

        attn_patterns = cache[get_act_name("attn", layer)]

        head_patterns = attn_patterns[:, head, :, :]

        score = 0
        for pos in range(seq_len, 2 * seq_len - 1):
            target_pos = pos - seq_len + 1  
            score += head_patterns[0, pos, target_pos].item()
        total_score += score / (seq_len - 1)

    return total_score / num_samples 

induction_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        score = calculate_induction_score(head, layer, num_samples=100)
        induction_scores[layer, head] = score
        print(f"Layer {layer} Head {head}: {score:.4f}")

flat_scores = induction_scores.flatten()
top_k = 10  
top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]

print("\nTop induction heads:")
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    print(f"Layer {layer} Head {head}: {flat_scores[idx]:.4f}")

def plot_attention_map(layer, head, seq_len=50):
    tokens = generate_synthetic_sequence(seq_len=seq_len)
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: "attn" in name,
        return_type=None
    )
    attn_patterns = cache[get_act_name("attn", layer)]
    head_attn = attn_patterns[0, head, :, :].detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(head_attn, cmap="viridis")
    plt.title(f"Attention Map - Layer {layer} Head {head}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    filename = f"attention_map_layer{layer}_head{head}.png"
    if 'instruct' in model.lower():
        path = "llama_8b_instruct_ih/" + filename
    else:
        path = "llama_8b_ih/" + filename
    plt.savefig(path)
    plt.close()
    print(f"Saved {filename}")

for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    plot_attention_map(layer, head)
