import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def compute_metrics(eval_pred):
    """
    Computes accuracy based on the predicted token for the target.
    Since we append a dummy token and the model is causal, the prediction 
    for the target (located at index valid_index) comes from logits at valid_index - 1.
    """
    logits, labels = eval_pred
    pred_labels = []
    true_labels = []
    for logit, label in zip(logits, labels):
        valid_indices = np.where(label != -100)[0]
        valid_index = valid_indices[-1]  # target label is at this position.
        # Shift one back because of causal prediction
        pred_token = np.argmax(logit[valid_index - 1])
        true_token = label[valid_index]
        pred_labels.append(pred_token)
        true_labels.append(true_token)
    accuracy = np.mean(np.array(pred_labels) == np.array(true_labels))
    return {"accuracy": accuracy}



# Minimal dataset that returns (x, y) pairs.
class DiffXYDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# Generate a dataset based on simple functions.
def get_dataset(lang_params):
    sampling_type = lang_params['name']        # 'first', 'last', 'induction_left', or 'induction_right'
    min_len = lang_params['min_len']
    max_len = lang_params['max_len']
    total_samples = lang_params['total_samples']
    x_data, y_data = [], []
    
    if sampling_type in ['first', 'last']:
        for _ in range(total_samples):
            seq_len = torch.randint(min_len, max_len, (1,)).item()
            x = torch.randint(0, 5, (seq_len,))
            x_data.append(x)
            y = torch.tensor([x[0]]) if sampling_type == 'first' else torch.tensor([x[-1]])
            y_data.append(y)

    elif sampling_type in ['induction_left', 'induction_right']:
        for _ in range(total_samples):
            L = torch.randint(min_len, max_len, (1,)).item()
            while L < 4:
                L = torch.randint(min_len, max_len, (1,)).item()
            target = torch.randint(0, 5, (1,)).item()
            possible_tokens = list(range(5))
            possible_tokens.remove(target)
            body_length = L - 1
            body = [random.choice(possible_tokens) for _ in range(body_length)]
            insertion_index = random.randint(1, body_length - 1)
            body.insert(insertion_index, target)
            full_seq = torch.tensor(body + [target])
            label = full_seq[insertion_index - 1] if sampling_type == 'induction_left' else full_seq[insertion_index + 1]
            x_data.append(full_seq)
            y_data.append(torch.tensor([label]))
    
    return DiffXYDataset(x_data, y_data)


def data_collator(batch):
    pad_token_id = 5  # pad token remains 5.
    dummy_token_id = 6  # NEW dedicated token for the answer slot.
    # Each example now will have an extra token (dummy_token) appended for the answer.
    max_x_len = max(x.shape[0] for x, _ in batch)
    max_len = max_x_len + 1  # extra token for the answer slot

    input_ids, labels, attention_mask = [], [], []
    for x, y in batch:
        orig_len = x.shape[0]
        # Append the dummy token at the end of the input.
        # new_x = torch.cat([x, torch.tensor([dummy_token_id], dtype=torch.long)])
        # Pad the extended sequence to the maximum length.
        padding = torch.full((max_len - orig_len,), pad_token_id, dtype=torch.long)
        padded_x = torch.cat([x, padding])
        input_ids.append(padded_x)
        
        # Create an attention mask for the actual tokens (including the answer slot).
        attn_mask = torch.cat([torch.ones(orig_len, dtype=torch.long),
                                 torch.zeros(max_len - orig_len, dtype=torch.long)])
        attention_mask.append(attn_mask)
        
        # Create labels: set -100 everywhere except at the answer slot.
        label_seq = torch.full((max_len,), -100, dtype=torch.long)
        # The answer should be predicted at the appended dummy position (index = orig_len)
        label_seq[orig_len] = y.item()
        # print("X", padded_x, "attn_mask", attn_mask, "label_seq", label_seq)
        labels.append(label_seq)
    
    return {    
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }


def get_or_create_dataset(lang_params, dataset_type: str):
    """
    Checks if a cached dataset file exists for the given dataset_type ("train" or "val").
    If so, loads it; otherwise, generates the dataset and saves it locally.
    """
    cache_path = (
        f"datasets/cached_{dataset_type}_{lang_params['name']}_"
        f"min{lang_params['min_len']}_max{lang_params['max_len']}_"
        f"samples{lang_params['total_samples']}.pt"
    )
    if os.path.exists(cache_path):
        torch.serialization.add_safe_globals([DiffXYDataset])
        dataset = torch.load(cache_path, weights_only=True)
        print(f"Loaded {dataset_type} dataset from {cache_path}")
        return dataset
    else:
        dataset = get_dataset(lang_params)
        torch.save(dataset, cache_path)
        print(f"Saved {dataset_type} dataset to {cache_path}")
        return dataset