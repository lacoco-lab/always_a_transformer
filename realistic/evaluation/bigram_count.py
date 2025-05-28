import os
import jsonlines
from typing import List, Dict
from collections import defaultdict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

HF_TOKEN = os.environ.get("HF_TOKEN", None)

def get_tokenizer_for_model(model_name: str):
    model_name = model_name.lower()
    model_to_hf = {
        "qwen2.5_7b": "Qwen/Qwen2.5-7B",
        "qwen2.5_32b": "Qwen/Qwen2.5-32B",
        "qwen2.5_7b_instruct": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5_32b_instruct": "Qwen/Qwen2.5-32B-Instruct",
        "llama3_8b": "meta-llama/Llama-3.1-8B",
        "llama3_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3_70b": "meta-llama/Meta-Llama-3.1-70B",
        "llama3_70b_instruct": "meta-llama/Llama-3.3-70B-Instruct",
    }
    hf_name = model_to_hf.get(model_name)
    if not hf_name:
        raise ValueError(f"Unknown model: {model_name}")
    return AutoTokenizer.from_pretrained(hf_name, token=HF_TOKEN, trust_remote_code=True)

class BigramAnalyzer:
    @staticmethod
    def get_bigram_version(tokens: List[str]) -> List[tuple]:
        return list(zip(tokens, tokens[1:])) if len(tokens) >= 2 else []

    @staticmethod
    def categorize_bigrams(bigram_list: List[tuple]):
        first_to_next = defaultdict(set)
        for first, second in bigram_list:
            first_to_next[first].add(second)
        total_unique_tokens = first_to_next.keys()
        det = [tok for tok, nxt in first_to_next.items() if len(nxt) == 1]
        non_det = [tok for tok, nxt in first_to_next.items() if len(nxt) > 1]
        return total_unique_tokens, det, non_det

def analyze_paragraphs(tokenizer, paragraphs: List[str]) -> List[Dict[str,int]]:
    results = []
    for para in paragraphs:
        tokens = tokenizer.tokenize(para)
        bigrams = BigramAnalyzer.get_bigram_version(tokens)
        total, det, non_det = BigramAnalyzer.categorize_bigrams(bigrams)
        results.append({
            "total_tokens": len(total),
            "deterministic_bigrams": len(det),
            "non_deterministic_bigrams": len(non_det),
        })
    return results

if __name__ == "__main__":
    models = ['llama3_8b', 'llama3_70b', 'qwen2.5_7b', 'qwen2.5_32b']
    dataset_files = [
        'datasets/realistic/loremipsum/500_tokens_seed_0.jsonl',
        'datasets/realistic/loremipsum/500_tokens_seed_1.jsonl',
        'datasets/realistic/loremipsum/500_tokens_seed_2.jsonl'
    ]

    model_names = []
    avg_dets = []
    avg_non_dets = []
    avg_totals = []

    for model in models:
        tokenizer = get_tokenizer_for_model(model)
        sum_det = 0
        sum_non_det = 0
        sum_total = 0
        n_paras = 0

        for infile in dataset_files:
            with jsonlines.open(infile) as reader:
                paragraphs = [r['input'] for r in reader]
            stats = analyze_paragraphs(tokenizer, paragraphs)
            n_paras += len(stats)
            for s in stats:
                sum_det += s['deterministic_bigrams']
                sum_non_det += s['non_deterministic_bigrams']
                sum_total += s['total_tokens']

        # Compute averages
        avg_det = sum_det / n_paras if n_paras else 0
        avg_non_det = sum_non_det / n_paras if n_paras else 0
        avg_total = sum_total / n_paras if n_paras else 0

        # Store for plotting
        model_names.append(model)
        avg_dets.append(avg_det)
        avg_non_dets.append(avg_non_det)
        avg_totals.append(avg_total)

        # Print intermediate values
        print(f"Model {model}:")
        print(f"  Paragraphs processed:       {n_paras}")
        print(f"  Avg. deterministic: {avg_det:.2f}")
        print(f"  Avg. non-deterministic:     {avg_non_det:.2f}")
        print(f"  Avg. tokens:     {avg_total:.2f}")
        print()

    # --- Plotting ---
    x = list(range(len(model_names)))
    width = 0.35

    plt.figure()
    plt.bar([xi - width/2 for xi in x], avg_dets, width, label='Unambiguous')
    plt.bar([xi + width/2 for xi in x], avg_non_dets, width, label='Ambiguous')
    plt.xticks(x, model_names, rotation=0, ha='right')
    plt.ylabel('Average bigrams per paragraph')
    plt.title('Deterministic vs Non-deterministic Bigrams (avg per paragraph)')
    plt.legend()
    plt.tight_layout()
    plt.show()
