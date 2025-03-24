import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformer_lens import head_detector
import sys
from argparse import ArgumentParser
import os

# Argument parsing
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", help="choose model")
args = parser.parse_args()

# Model selection
if args.model == 'llama3.1-8b-instruct':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    version = 'instruct'
    save_dir = "llama3_8b_instruct_ih"
else:
    model_name = "meta-llama/Meta-Llama-3-8B"
    version = 'non-instruct'
    save_dir = "llama3_8b_ih"

os.makedirs(save_dir, exist_ok=True)

# Load model
device = "cuda" if torch.cuda.is_available() else "mps"
model = HookedTransformer.from_pretrained(model_name, device=device)

# Get induction scores
print("Calculating induction scores...")
induction_scores = head_detector.induction_score(
    model,
    seq_len=50,
    batch_size=10,
    prepend_bos=False
)

# Flatten and get top K
flat_scores = induction_scores.flatten()
top_k = 10
top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]

print("\nTop induction heads:")
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    print(f"Layer {layer} Head {head}: {flat_scores[idx]:.4f}")

# Plot: Induction Scores Heatmap
def plot_induction_scores_heatmap(scores, save_path):
    plt.figure(figsize=(12, 6))
    plt.imshow(scores, cmap="viridis")
    plt.colorbar(label="Induction Score")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Induction Scores Heatmap")
    plt.xticks(np.arange(model.cfg.n_heads))
    plt.yticks(np.arange(model.cfg.n_layers))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")

plot_induction_scores_heatmap(
    induction_scores,
    os.path.join(save_dir, "induction_scores_heatmap.png")
)

# Plot attention maps for top heads
def plot_attention_map(model, layer, head, seq_len=50):
    tokens = head_detector.generate_induction_token_sequence(
        model, seq_len=seq_len, prepend_bos=False
    )
    _, cache = model.run_with_cache(tokens)
    attn = cache["attn"]  # [batch, head, q, k]
    head_attn = attn[0, head].detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(head_attn, cmap="viridis")
    plt.title(f"Attention Map - Layer {layer} Head {head}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    filename = f"attention_map_layer{layer}_head{head}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved {filename}")

# Visualize top heads
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    plot_attention_map(model, layer, head)
