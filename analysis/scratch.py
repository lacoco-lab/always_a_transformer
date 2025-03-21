from collections import defaultdict
from typing import List, Tuple, Dict, Any


def get_bigram_version(text_list: List[Any]) -> List[Tuple[Any, Any]]:
    """
    Convert a list into a list of bigrams (tuples of adjacent elements).

    Args:
        text_list: List of elements to convert to bigrams

    Returns:
        List of tuples representing bigrams
    """
    return list(zip(text_list, text_list[1:]))


def categorize_bigrams(bigram_list: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any], Dict[Any, set]]:
    """
    Categorize first tokens into deterministic and non-deterministic based on possible continuations.

    Args:
        bigram_list: List of bigram tuples

    Returns:
        - List of deterministic first tokens
        - List of non-deterministic first tokens
        - Dictionary mapping first tokens to their possible continuations
    """
    # Create mapping of first token to all possible continuations
    first_dict = defaultdict(set)
    for first, second in bigram_list:
        first_dict[first].add(second)

    # Categorize tokens based on number of continuations
    deterministic = []
    non_deterministic = []

    # Iterate over first_dict to avoid duplicates
    for first, seconds in first_dict.items():
        if len(seconds) == 1:
            deterministic.append(first)
        else:
            non_deterministic.append(first)

    return deterministic, non_deterministic, first_dict


def levenshtein_distance(seq1, seq2):
    """
    Calculate the Levenshtein distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        The Levenshtein distance as an integer
    """
    m, n = len(seq1), len(seq2)

    # Create a matrix of size (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                   dp[i][j - 1],  # insertion
                                   dp[i - 1][j - 1])  # substitution

    return dp[m][n]


def get_bigram_accuracy(i_bigram_v: List[Tuple[Any, Any]], o_bigram_v: List[Tuple[Any, Any]]) -> Dict[str, float]:
    """
    Calculate Levenshtein-based metrics for comparing bigram sequences.

    Args:
        i_bigram_v: Input (reference) bigram list
        o_bigram_v: Output (comparison) bigram list

    Returns:
        Dictionary with Levenshtein similarity scores for overall, deterministic, and non-deterministic bigrams
    """
    # Identify deterministic and non-deterministic first tokens
    det_firsts, non_det_firsts, _ = categorize_bigrams(i_bigram_v)

    # Extract deterministic and non-deterministic bigrams from input sequence
    det_bigrams_i = [bg for bg in i_bigram_v if bg[0] in det_firsts]
    non_det_bigrams_i = [bg for bg in i_bigram_v if bg[0] in non_det_firsts]

    # Extract corresponding bigram types from output sequence
    det_bigrams_o = [bg for bg in o_bigram_v if bg[0] in det_firsts]
    non_det_bigrams_o = [bg for bg in o_bigram_v if bg[0] in non_det_firsts]

    # Calculate Levenshtein distances
    overall_distance = levenshtein_distance(i_bigram_v, o_bigram_v)
    det_distance = levenshtein_distance(det_bigrams_i, det_bigrams_o)
    non_det_distance = levenshtein_distance(non_det_bigrams_i, non_det_bigrams_o)

    # Calculate max possible distances for normalization
    max_overall_distance = max(len(i_bigram_v), len(o_bigram_v))
    max_det_distance = max(len(det_bigrams_i), len(det_bigrams_o))
    max_non_det_distance = max(len(non_det_bigrams_i), len(non_det_bigrams_o))

    # Normalize to similarity scores (1 - distance/max_distance)
    # Handle edge case of empty sequences
    overall_similarity = 1 - (overall_distance / max_overall_distance if max_overall_distance > 0 else 0)
    det_similarity = 1 - (det_distance / max_det_distance if max_det_distance > 0 else 0)
    non_det_similarity = 1 - (non_det_distance / max_non_det_distance if max_non_det_distance > 0 else 0)

    # Additional metrics to help understand the results better
    metrics = {
        "overall": {
            "levenshtein_similarity": overall_similarity,
            "input_count": len(i_bigram_v),
            "output_count": len(o_bigram_v),
            "edit_distance": overall_distance
        },
        "deterministic": {
            "levenshtein_similarity": det_similarity,
            "input_count": len(det_bigrams_i),
            "output_count": len(det_bigrams_o),
            "edit_distance": det_distance
        },
        "non_deterministic": {
            "levenshtein_similarity": non_det_similarity,
            "input_count": len(non_det_bigrams_i),
            "output_count": len(non_det_bigrams_o),
            "edit_distance": non_det_distance
        }
    }

    return metrics


def format_metrics(metrics: Dict) -> str:
    """
    Format the metrics dictionary into a readable string.

    Args:
        metrics: Dictionary of metrics from get_bigram_accuracy

    Returns:
        Formatted string representation of metrics
    """
    result = []
    result.append("=== BIGRAM COMPARISON METRICS (LEVENSHTEIN) ===")

    # Overall metrics
    overall = metrics["overall"]
    result.append("\nOverall:")
    result.append(f"  Levenshtein Similarity: {overall['levenshtein_similarity']:.4f} (1.0 = identical)")
    result.append(f"  Edit Distance: {overall['edit_distance']} operations")
    result.append(f"  Input Bigrams: {overall['input_count']}")
    result.append(f"  Output Bigrams: {overall['output_count']}")

    # Deterministic metrics
    det = metrics["deterministic"]
    result.append("\nDeterministic Bigrams:")
    result.append(f"  Levenshtein Similarity: {det['levenshtein_similarity']:.4f} (1.0 = identical)")
    result.append(f"  Edit Distance: {det['edit_distance']} operations")
    result.append(f"  Input Bigrams: {det['input_count']}")
    result.append(f"  Output Bigrams: {det['output_count']}")

    # Non-deterministic metrics
    non_det = metrics["non_deterministic"]
    result.append("\nNon-Deterministic Bigrams:")
    result.append(f"  Levenshtein Similarity: {non_det['levenshtein_similarity']:.4f} (1.0 = identical)")
    result.append(f"  Edit Distance: {non_det['edit_distance']} operations")
    result.append(f"  Input Bigrams: {non_det['input_count']}")
    result.append(f"  Output Bigrams: {non_det['output_count']}")

    return "\n".join(result)


if __name__ == "__main__":
    # Example usage
    sentence = ['a', 'b', 'c', 'c', 'a', 'f', 'b', 'c']
    cases = [
        ['a', 'b', 'c', 'a', 'f', 'b', 'c'],
    ]

    # Display reference information
    i_bigram_v = get_bigram_version(sentence)
    det_firsts, non_det_firsts, first_dict = categorize_bigrams(i_bigram_v)

    print(f"Reference sequence: {sentence}")
    print(f"Reference bigrams: {i_bigram_v}")

    print("\nFirst token categorization:")
    for first, continuations in first_dict.items():
        print(f"  '{first}' → {continuations} ({'deterministic' if len(continuations) == 1 else 'non-deterministic'})")

    print(f"\nDeterministic first tokens: {det_firsts}")
    print(f"Non-deterministic first tokens: {non_det_firsts}")

    # Process each test case
    for i, case in enumerate(cases):
        print(f"\n\n--- Case {i + 1}: {case} ---")
        o_bigram_v = get_bigram_version(case)
        print(f"Case bigrams: {o_bigram_v}")

        # Compare using Levenshtein metrics
        metrics = get_bigram_accuracy(i_bigram_v, o_bigram_v)
        print(format_metrics(metrics))