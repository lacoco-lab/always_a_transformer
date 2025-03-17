import nltk
import re
from collections import defaultdict

def clean_text(sentence):
    return re.sub(r'[^\w\s]', '', sentence).replace('\n', ' ').replace('\r', '')

def classify_bigrams(original_bigrams):
    """Classify bigrams into unique and non-unique based on the first token's predictability."""
    first_token_map = defaultdict(set)

    for first, second in original_bigrams:
        first_token_map[first].add(second)

    unique_bigrams = {bigram for bigram in original_bigrams if len(first_token_map[bigram[0]]) == 1}
    non_unique_bigrams = {bigram for bigram in original_bigrams if len(first_token_map[bigram[0]]) > 1}
    return unique_bigrams, non_unique_bigrams

def pad_ans(gold, ans):
    """Pad the copied sequence if it is shorter than the original sequence."""
    if len(ans.split()) < len(gold.split()):
        while len(ans.split()) != len(gold.split()):
            ans += " <pad>"
    return ans

def calculate_copy_accuracy(original_bigrams, copied_bigrams):
    """Calculate copy accuracy for unique and non-unique bigrams while skipping hallucinated bigrams."""
    unique_bigrams, non_unique_bigrams = classify_bigrams(original_bigrams)

    correct_unique = 0
    correct_non_unique = 0

    total_unique = sum(1 for bigram in original_bigrams if bigram in unique_bigrams)
    total_non_unique = sum(1 for bigram in original_bigrams if bigram in non_unique_bigrams)

    orig_idx, copied_idx = 0, 0  # Two pointers for original and copied bigrams

    while orig_idx < len(original_bigrams) and copied_idx < len(copied_bigrams):
        orig_bigram = original_bigrams[orig_idx]
        copied_bigram = copied_bigrams[copied_idx]

        if copied_bigram == orig_bigram:  # Correct match
            if orig_bigram in unique_bigrams:
                correct_unique += 1
            elif orig_bigram in non_unique_bigrams:
                correct_non_unique += 1
            orig_idx += 1  # Move both pointers forward
            copied_idx += 1
        elif copied_bigram not in original_bigrams:  # Hallucinated bigram
            copied_idx += 1  # Skip this bigram in copied sequence
        else:
            orig_idx += 1  # Move forward in original sequence

    unique_accuracy = correct_unique / total_unique if total_unique > 0 else 0
    non_unique_accuracy = correct_non_unique / total_non_unique if total_non_unique > 0 else 0

    return unique_accuracy, non_unique_accuracy

gold = "Adipisci dolor velit neque dolorem ut sit velit porro."
ans1 = "Adipisci dolor velit neque dolorem ut sit velit porro."

gold = clean_text(gold)
ans1 = clean_text(ans1)

gold_bigrams = list(nltk.bigrams(gold.split()))
ans1 = pad_ans(gold, ans1)
print(ans1)

ans_bigrams = list(nltk.bigrams(ans1.split()))
print(calculate_copy_accuracy(gold_bigrams, ans_bigrams))
