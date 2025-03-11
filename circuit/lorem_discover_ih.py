import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import matplotlib.pyplot as plt

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def get_data(path):
    """
    Read jsonl file with the results.
    :param path: path of the jsonl file
    :return: list of dicts with results
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


# Dataset definition
class CopyTaskDataset:
    def __init__(self, paragraphs, instruction="Repeat the paragraph above exactly as it is, without any changes\n"):
        self.paragraphs = paragraphs
        self.instruction = instruction

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        full_prompt = f"{self.instruction}{self.paragraphs[idx]}\nCopy:"
        return {
            "text": self.paragraphs[idx],
            "prompt": full_prompt,
            "tokens": tokenizer.encode(full_prompt, return_tensors="pt")[0]  # tensor shape: [seq_len]
        }


def collate_fn(batch):
    texts = [item["text"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    tokens_list = [item["tokens"] for item in batch]
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {"text": texts, "prompt": prompts, "tokens": padded_tokens}


def calculate_accuracy(predictions, targets):
    total = 0
    correct = 0
    for pred, target in zip(predictions, targets):
        print(pred)
        print(target)
        if pred.strip() == target.strip():
            correct += 1
        total += 1
    return correct / total


def analyze_induction_heads_batch(dataloader, num_samples=1500):
    """
    Perform batch analysis with sampling settings for generation.

    :param dataloader: DataLoader yielding batches.
    :param num_samples: Maximum number of samples to process.
    :param temperature: Temperature for sampling.
    :param top_k: Top-k parameter for sampling.
    :return: (induction_scores, accuracy)
    """
    induction_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    all_predictions = []
    all_targets = []
    sample_count = 0

    for batch in tqdm(dataloader):
        tokens = batch["tokens"].to(device)  # shape: [batch_size, seq_len]
        batch_size = tokens.size(0)

        with torch.no_grad():
            logits = model(tokens)  # shape: [batch_size, seq_len, vocab_size]
            predictions = torch.argmax(logits[:, -1], dim=-1)


        for i in range(batch_size):
            generated = tokenizer.decode(predictions[i].cpu().numpy())
            all_predictions.append(generated)
            all_targets.append(batch["text"][i])


        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "attn" in name,
            return_type=None
        )

        # Compute induction head scores.
        for i in range(batch_size):
            seq_tokens = tokens[i]  # [seq_len]
            valid_length = (
                        seq_tokens != tokenizer.pad_token_id).sum().item() if tokenizer.pad_token_id is not None else seq_tokens.size(
                0)
            for layer in range(model.cfg.n_layers):
                patterns = cache[f"blocks.{layer}.attn.hook_pattern"][i]
                for head in range(model.cfg.n_heads):
                    head_score = 0.0
                    valid_positions = 0
                    for q_pos in range(1, valid_length):
                        current_token = seq_tokens[q_pos].item()
                        prev_occurrences = (seq_tokens[:q_pos] == current_token).nonzero(as_tuple=False)
                        if prev_occurrences.numel() == 0:
                            continue
                        k_pos = prev_occurrences[-1].item()
                        head_score += patterns[head, q_pos, k_pos].item()
                        valid_positions += 1
                    if valid_positions > 0:
                        induction_scores[layer, head] += head_score / valid_positions

        sample_count += batch_size
        if sample_count >= num_samples:
            break

    induction_scores /= sample_count
    accuracy = calculate_accuracy(all_predictions, all_targets)
    return induction_scores, accuracy


def plot_attention_map_for_batch(batch, layer, head):
    """
    Plot the averaged attention map for a given batch, layer, and head.

    :param batch: A batch dictionary from the DataLoader (output of collate_fn).
    :param layer: The layer index for which to plot the attention map.
    :param head: The head index for which to plot the attention map.
    """
    tokens = batch["tokens"].to(device)  # shape: [batch_size, seq_len]
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: "attn" in name,
        return_type=None
    )

    # Extract attention maps: shape [batch_size, n_heads, seq_len, seq_len]
    attn_maps = cache[f"blocks.{layer}.attn.hook_pattern"]
    
    # Select the specific head: [batch_size, seq_len, seq_len]
    attn_maps_head = attn_maps[:, head, :, :]
    avg_attn_map = attn_maps_head.mean(dim=0).detach().cpu().numpy()

    token_ids = tokens[0]
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)

    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attn_map, cmap='viridis', aspect='auto')
    plt.colorbar(label='Average Attention Weight')
    plt.xticks(ticks=np.arange(len(token_texts)), labels=token_texts, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(token_texts)), labels=token_texts, fontsize=8)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title(f'Average Attention Map for Batch - Layer {layer} Head {head}')
    plt.tight_layout()
    filename = f"attention_map_layer{layer}_head{head}.png"
    path = "llama_8b_instruct_ih/" + "lorem/" + filename
    plt.savefig(path)
    plt.close()
    print(f"Saved {filename}")
    plt.show()


data = get_data('../datasets/500/loremipsum/data.jsonl')
paragraphs = [par["input"] for par in data][:1]

dataset = CopyTaskDataset(paragraphs)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn) 


induction_scores, accuracy = analyze_induction_heads_batch(dataloader)
print(f"Copy Task Accuracy: {accuracy:.2%}")
print("Top Induction Heads:")
flat_scores = induction_scores.flatten()
top_indices = np.argpartition(flat_scores, -10)[-10:]
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    batch = next(iter(dataloader))
    plot_attention_map_for_batch(batch, layer=layer, head=head)
    print(f"Layer {layer} Head {head}: {flat_scores[idx]:.4f}")

