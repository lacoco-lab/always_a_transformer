from collections import defaultdict, Counter
import nltk


def get_bigram_counts(bigrams):
    """
    Get counts of each bigram in the sentence.
    :param bigrams: arr 
    :return: dict
    """
    bigram_counts = Counter(bigrams)

    return bigram_counts


def categorize_bigrams(text: str):
    """
    Categorize bigrams into deterministic and non-deterministic based on the second token in the bigram.
    
    :param text: str, word-level input
    :return: arr, arr, arr
    """
    bigrams = list(nltk.bigrams(text.strip().replace(".", "").split()))

    first_dict = defaultdict(set)
    for first, second in bigrams:
        first_dict[first].add(second)
    deterministic = []
    non_deterministic = []

    for (first, second) in bigrams:
        if len(first_dict[first]) == 1:
            deterministic.append((first, second))
        else:
            non_deterministic.append((first, second))

    return bigrams, deterministic, non_deterministic


def pad_ans(gold, ans):
    """
    Pad the copied sequence if it is shorter than the original sequence.
    :param gold: str
    :param ans: str
    :return: str
    """
    if len(ans.split()) < len(gold.split()):
        while len(ans.split()) != len(gold.split()):
            ans += " <pad>"
    return ans


def get_bigram_accuracy(inp, out):
    """
    Get DT and Non-DT bigram accuracy.
    :param inp: str, gold sentence
    :param out: str, copied sentence
    :return: float, float
    """
    
    """
    Welcome to the cumbersome logic of bigram accuracies!
    There are many edge cases: hallucinated tokens, removed tokens, etc.
    I hope we are handling all of them. But maybe not!
    """
    
    orig_bigrams, dt, non_dt = categorize_bigrams(inp)
    out = pad_ans(inp, out) # check if the length is shorter and pad
    cp_bigrams = list(nltk.bigrams(out.strip().replace(".", "").split()))
    
    bigram_counts = get_bigram_counts(orig_bigrams)
    dt_bigram_accs = dict.fromkeys(dt, 0)
    non_dt_bigram_accs = dict.fromkeys(non_dt, 0)

    orig_idx, copied_idx = 0, 0
    while orig_idx < len(orig_bigrams) and copied_idx < len(cp_bigrams):
        o_bigram = orig_bigrams[orig_idx]
        c_bigram = cp_bigrams[copied_idx]
        print(o_bigram, c_bigram)

        if c_bigram[0] == o_bigram[0]:  # First token must match
            if c_bigram[1] == o_bigram[1]:  # Correct copy of second token
                target_dict = dt_bigram_accs if o_bigram in dt else non_dt_bigram_accs if o_bigram in non_dt else None
                if target_dict is not None:
                    target_dict[o_bigram] += 1
                orig_idx += 1  # Move forward in original bigrams
                copied_idx += 1  # Move forward in copied bigrams

            elif c_bigram[1] != o_bigram[1]:  # Mismatch in second token
                orig_idx += 1 

        elif o_bigram[0] != c_bigram[0]:  # First token mismatch
            if o_bigram[1] == c_bigram[1]: # Second match
                copied_idx += 1
            orig_idx += 1

    for key_dt, key_non_dt in zip(dt_bigram_accs, non_dt_bigram_accs):
        dt_bigram_accs[key_dt] = dt_bigram_accs[key_dt] / bigram_counts[key_dt]
        non_dt_bigram_accs[key_non_dt] = non_dt_bigram_accs[key_non_dt] / bigram_counts[key_non_dt]
    print(dt_bigram_accs)
    print(non_dt_bigram_accs)
    return sum(dt_bigram_accs.values()) / len(dt_bigram_accs), sum(non_dt_bigram_accs.values()) / len(non_dt_bigram_accs)


if __name__ == "__main__":
    sentence = "a b c c a f b c"
    copy = "a c c a f b c"
    bigrams, dt, non_dt = categorize_bigrams(sentence)
    #print("All bigrams: ", bigrams)
    #print("Copy bigrams: ", list(nltk.bigrams(copy.strip().replace(".", "").split())))
    #print("Deterministic bigrams:", dt)
    #print("Non-deterministic bigrams:", non_dt)
    dt_acc, non_dt_acc = get_bigram_accuracy(sentence, copy)
    #print(f"DT: {dt_acc}, Non-DT: {non_dt_acc}")