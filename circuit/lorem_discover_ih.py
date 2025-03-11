import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json

# Load model and tokenizer
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# If your tokenizer does not have a pad token, consider setting it:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def get_data(path):
    """
    Read jsonl file with the results.

    :param path: dir of the jsonl file
    :return: arr of dicts with results
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Dataset definition
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
            "tokens": tokenizer.encode(full_prompt, return_tensors="pt")[0]  # tensor shape: [seq_len]
        }


# Collate function to pad sequences in a batch
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    tokens_list = [item["tokens"] for item in batch]
    # pad_sequence pads a list of tensors to the length of the longest one in the batch.
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {"text": texts, "prompt": prompts, "tokens": padded_tokens}


# Accuracy calculation remains unchanged
def calculate_accuracy(predictions, targets):
    total = 0
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
        total += 1
    return correct / total


# Batch version of the induction head analysis
def analyze_induction_heads_batch(dataloader, num_samples=1500):
    induction_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    all_predictions = []
    all_targets = []
    sample_count = 0

    for batch in tqdm(dataloader):
        tokens = batch["tokens"].to(device)  # shape: [batch_size, seq_len]
        batch_size = tokens.size(0)

        # Run forward pass without cache to obtain predictions.
        with torch.no_grad():
            logits = model(tokens)  # logits shape: [batch_size, seq_len, vocab_size]
            # Get the prediction for the last token in each sample.
            predictions = torch.argmax(logits[:, -1], dim=-1)

        # Decode predictions for each sample in the batch.
        for i in range(batch_size):
            generated = tokenizer.decode(predictions[i].cpu().numpy())
            all_predictions.append(generated)
            all_targets.append(batch["text"][i])

        # Run forward pass with cache to capture attention patterns.
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "attn" in name,
            return_type=None
        )

        # Compute induction head scores for each sample in the batch.
        for i in range(batch_size):
            seq_tokens = tokens[i]  # [seq_len]
            # Determine the effective length by ignoring padded tokens.
            if tokenizer.pad_token_id is not None:
                valid_length = (seq_tokens != tokenizer.pad_token_id).sum().item()
            else:
                valid_length = seq_tokens.size(0)

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


data = get_data('../datasets/500/loremipsum/data.jsonl')
paragraphs = []
for par in data:
    paragraphs.append(par["input"])

dataset = CopyTaskDataset(paragraphs)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

induction_scores, accuracy = analyze_induction_heads_batch(dataloader)

print(f"Copy Task Accuracy: {accuracy:.2%}")
print("Top Induction Heads:")
flat_scores = induction_scores.flatten()
top_indices = np.argpartition(flat_scores, -10)[-10:]
for idx in top_indices:
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    print(f"Layer {layer} Head {head}: {flat_scores[idx]:.4f}")
