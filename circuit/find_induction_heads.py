import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformer_lens.utils import set_seed, get_act_name
from transformer_lens.head_detector import detect_head
import os
from argparse import ArgumentParser

# ----------------------------- #
#       CUSTOM GENERATOR       #
# ----------------------------- #
def generate_repeated_random_tokens(model, batch=1000, seq_len=50, seed=0):
    set_seed(seed)
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    rep_tokens = torch.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(model.cfg.device)
    return rep_tokens

# ----------------------------- #
#     DETECT INDUCTION HEADS   #
# ----------------------------- #
def find_random_induction_heads(model, batch=100, seq_len=50, seed=0):
    rep_tokens = generate_repeated_random_tokens(model, batch, seq_len, seed)
    prompts = [model.tokenizer.decode(toks.tolist()) for toks in rep_tokens]

    head_scores = detect_head(
        model,
        prompts=prompts,
        task="induction_head",
        exclude_bos=False,
        exclude_current_token=False,
        error_measure="abs"
    )

    results = dict()
    for layer, layer_scores in enumerate(head_scores):
        for head, score in enumerate(layer_scores):
            results[f'{layer}.{head}'] = score.item()
    return head_scores, results

# ----------------------------- #
#         HEATMAP PLOT         #
# ----------------------------- #
def plot_induction_scores_heatmap(scores, save_path):
    plt.figure(figsize=(12, 6))
    plt.imshow(scores, cmap="viridis")
    plt.colorbar(label="Induction Score")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Induction Scores Heatmap")
    plt.xticks(np.arange(scores.shape[1]))
    plt.yticks(np.arange(scores.shape[0]))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")

# ----------------------------- #
#      ATTENTION MAP PLOTS     #
# ----------------------------- #
def plot_attention_map(model, layer, head, seq_len, out_dir):
    tokens = generate_repeated_random_tokens(model, batch=1, seq_len=seq_len)
    _, cache = model.run_with_cache(tokens)
    attn = cache[get_act_name("attn", layer)]
    head_attn = attn[0, head].detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(head_attn, cmap="viridis")
    plt.title(f"Attention Map - Layer {layer} Head {head}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    filename = f"attention_map_layer{layer}_head{head}.png"
    path = os.path.join(out_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved attention map: {path}")

# ----------------------------- #
#              MAIN            #
# ----------------------------- #
def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="LLaMA model variant", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--out_dir", help="Directory to save outputs", default="llama_induction_out")
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Detecting induction heads...")
    head_scores, _ = find_random_induction_heads(model, batch=args.batch, seq_len=args.seq_len, seed=42)

    # Save heatmap
    heatmap_path = os.path.join(args.out_dir, "induction_scores_heatmap.png")
    head_scores_tensor = torch.tensor(head_scores)
    plot_induction_scores_heatmap(head_scores_tensor, heatmap_path)

    # Plot attention maps for top K
    flat_scores = head_scores_tensor.flatten()
    top_indices = torch.topk(flat_scores, args.top_k).indices.tolist()
    print(f"\nTop {args.top_k} Induction Heads:")
    for idx in top_indices:
        layer, head = divmod(idx, model.cfg.n_heads)
        score = head_scores_tensor[layer, head].item()
        print(f"Layer {layer} Head {head}: {score:.4f}")
        plot_attention_map(model, layer, head, seq_len=args.seq_len, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
