import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class CopyTaskDataset:
    def __init__(self, paragraphs, instruction="Please copy this text exactly:\n"):
        self.paragraphs = paragraphs
        self.instruction = instruction

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        full_prompt = f"{self.instruction}{self.paragraphs[idx]}\nCopy:"
        return {
            "text": self.paragraphs[idx],
            "prompt": full_prompt,
            "tokens": tokenizer.encode(full_prompt, return_tensors="pt")[0]
        }


def calculate_accuracy(predictions, targets):
    total = 0
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
        total += 1
    return correct / total


def analyze_induction_heads(dataset, num_samples=1500):
    induction_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    all_predictions = []
    all_targets = []

    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        targets = sample["text"]
        tokens = sample["tokens"].to(device)

        with torch.no_grad():
            logits = model(tokens.unsqueeze(0))
            predictions = torch.argmax(logits[:, -1], dim=-1)
            generated = tokenizer.decode(predictions[0].cpu().numpy())
            all_predictions.append(generated)
            all_targets.append(targets)

        _, cache = model.run_with_cache(
            tokens.unsqueeze(0),
            names_filter=lambda name: "attn" in name,
            return_type=None
        )

        seq_len = tokens.shape[0]
        for layer in range(model.cfg.n_layers):
            patterns = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [head, query, key]

            for head in range(model.cfg.n_heads):
                head_score = 0
            valid_positions = 0

            for q_pos in range(1, seq_len):
                current_token = tokens[q_pos].item()

            prev_occurrences = (tokens[:q_pos] == current_token).nonzero()
            if prev_occurrences.nelement() == 0:
                continue

            k_pos = prev_occurrences[-1].item()
            head_score += patterns[head, q_pos, k_pos].item()
            valid_positions += 1

        if valid_positions > 0:
            induction_scores[layer, head] += head_score / valid_positions


    induction_scores /= num_samples
    accuracy = calculate_accuracy(all_predictions, all_targets)
    
    return induction_scores, accuracy


paragraphs = [...]  
dataset = CopyTaskDataset(paragraphs)

induction_scores, accuracy = analyze_induction_heads(dataset)

print(f"Copy Task Accuracy: {accuracy:.2%}")
print("Top Induction Heads:")
flat_scores = induction_scores.flatten()
top_indices = np.argpartition(flat_scores, -10)[-10:]  # Top 10 heads
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    print(f"Layer {layer} Head {head}: {flat_scores[idx]:.4f}")