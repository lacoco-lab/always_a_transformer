from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union


class BigramComparator:
    """
    A utility class for comparing token sequences using bigram analysis.
    """
    
    @staticmethod
    def get_bigram_version(tokens: List[Any]) -> List[Tuple[Any, Any]]:
        """
        Convert a list of tokens into a list of bigrams.
        
        Args:
            tokens: List of tokens to convert to bigrams
            
        Returns:
            List of bigram tuples
        """
        if len(tokens) < 2:
            return []
        return list(zip(tokens, tokens[1:]))
    
    @staticmethod
    def categorize_bigrams(bigram_list: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any], Dict[Any, set]]:
        """
        Categorize first tokens as deterministic or non-deterministic based on continuations.
        
        Args:
            bigram_list: List of bigram tuples
            
        Returns:
            - List of deterministic first tokens
            - List of non-deterministic first tokens
            - Dictionary mapping first tokens to possible continuations
        """
        first_dict = defaultdict(set)
        for first, second in bigram_list:
            first_dict[first].add(second)
        
        deterministic = []
        non_deterministic = []
        
        for first, seconds in first_dict.items():
            if len(seconds) == 1:
                deterministic.append(first)
            else:
                non_deterministic.append(first)
        
        print("DETERMINISTIC -- ", deterministic, "NON_DETERMINISTIC -- ", non_deterministic)
        return deterministic, non_deterministic, first_dict
    
    @staticmethod
    def levenshtein_distance(seq1: List, seq2: List) -> int:
        """
        Calculate the Levenshtein distance between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            The Levenshtein distance as an integer
        """
        if not seq1:
            return len(seq2)
        if not seq2:
            return len(seq1)
            
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
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                       dp[i][j-1],      # insertion
                                       dp[i-1][j-1])    # substitution
        
        return dp[m][n]
    
    @classmethod
    def compare_token_sequences(cls, 
                               reference_tokens: List[Any], 
                               hypothesis_tokens: List[Any],
                               return_formatted: bool = False) -> Union[Dict, str]:
        """
        Compare two token sequences using bigram analysis and Levenshtein distance.
        
        Args:
            reference_tokens: Reference sequence of tokens
            hypothesis_tokens: Hypothesis sequence of tokens to compare against reference
            return_formatted: If True, returns a formatted string instead of metrics dictionary
            
        Returns:
            Dictionary with detailed metrics or formatted string if return_formatted=True
        """
        # Convert token sequences to bigrams
        ref_bigrams = cls.get_bigram_version(reference_tokens)
        hyp_bigrams = cls.get_bigram_version(hypothesis_tokens)
        
        # Skip calculation if one sequence is too short
        if not ref_bigrams or not hyp_bigrams:
            empty_metrics = {
                "overall": {
                    "levenshtein_similarity": 0.0 if not ref_bigrams or not hyp_bigrams else 1.0,
                    "input_count": len(ref_bigrams),
                    "output_count": len(hyp_bigrams),
                    "edit_distance": max(len(ref_bigrams), len(hyp_bigrams))
                },
                "deterministic": {
                    "levenshtein_similarity": 0.0,
                    "input_count": 0,
                    "output_count": 0,
                    "edit_distance": 0
                },
                "non_deterministic": {
                    "levenshtein_similarity": 0.0,
                    "input_count": 0,
                    "output_count": 0,
                    "edit_distance": 0
                }
            }
            
            if return_formatted:
                return cls.format_metrics(empty_metrics)
            return empty_metrics
        
        # Identify deterministic and non-deterministic tokens in reference
        det_firsts, non_det_firsts, _ = cls.categorize_bigrams(ref_bigrams)
        
        # Extract bigram subsets
        det_bigrams_ref = [bg for bg in ref_bigrams if bg[0] in det_firsts]
        non_det_bigrams_ref = [bg for bg in ref_bigrams if bg[0] in non_det_firsts]
        
        det_bigrams_hyp = [bg for bg in hyp_bigrams if bg[0] in det_firsts]
        non_det_bigrams_hyp = [bg for bg in hyp_bigrams if bg[0] in non_det_firsts]
        
        print(det_bigrams_ref, det_bigrams_hyp)
        # Calculate Levenshtein distances
        overall_distance = cls.levenshtein_distance(ref_bigrams, hyp_bigrams)
        det_distance = cls.levenshtein_distance(det_bigrams_ref, det_bigrams_hyp)
        non_det_distance = cls.levenshtein_distance(non_det_bigrams_ref, non_det_bigrams_hyp)
        
        # Calculate max possible distances for normalization
        max_overall_distance = max(len(ref_bigrams), len(hyp_bigrams))
        max_det_distance = max(len(det_bigrams_ref), len(det_bigrams_hyp))
        max_non_det_distance = max(len(non_det_bigrams_ref), len(non_det_bigrams_hyp))
        
        # Normalize to similarity scores (1 - distance/max_distance)
        overall_similarity = 1 - (overall_distance / max_overall_distance if max_overall_distance > 0 else 0)
        det_similarity = 1 - (det_distance / max_det_distance if max_det_distance > 0 else 0)
        non_det_similarity = 1 - (non_det_distance / max_non_det_distance if max_non_det_distance > 0 else 0)
        
        # Compile metrics
        metrics = {
            "overall": {
                "levenshtein_similarity": overall_similarity,
                "input_count": len(ref_bigrams),
                "output_count": len(hyp_bigrams),
                "edit_distance": overall_distance
            },
            "deterministic": {
                "levenshtein_similarity": det_similarity,
                "input_count": len(det_bigrams_ref),
                "output_count": len(det_bigrams_hyp),
                "edit_distance": det_distance
            },
            "non_deterministic": {
                "levenshtein_similarity": non_det_similarity,
                "input_count": len(non_det_bigrams_ref),
                "output_count": len(non_det_bigrams_hyp),
                "edit_distance": non_det_distance
            }
        }
        
        # Add token-level metrics
        token_level = {
            "reference_tokens": len(reference_tokens),
            "hypothesis_tokens": len(hypothesis_tokens),
            "token_match_ratio": len([i for i, (ref, hyp) in 
                                    enumerate(zip(reference_tokens, hypothesis_tokens)) 
                                    if ref == hyp]) / max(len(reference_tokens), len(hypothesis_tokens))
                                    if reference_tokens and hypothesis_tokens else 0.0
        }
        metrics["token_level"] = token_level
        
        if return_formatted:
            return cls.format_metrics(metrics)
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict) -> str:
        """
        Format the metrics dictionary into a readable string.
        
        Args:
            metrics: Dictionary of comparison metrics
            
        Returns:
            Formatted string representation of metrics
        """
        result = []
        result.append("=== BIGRAM COMPARISON METRICS (LEVENSHTEIN) ===")
        
        # Token-level metrics
        if "token_level" in metrics:
            tl = metrics["token_level"]
            result.append("\nToken-Level:")
            result.append(f"  Reference Tokens: {tl['reference_tokens']}")
            result.append(f"  Hypothesis Tokens: {tl['hypothesis_tokens']}")
            result.append(f"  Token Match Ratio: {tl['token_match_ratio']:.4f}")
        
        # Overall metrics
        overall = metrics["overall"]
        result.append("\nOverall Bigrams:")
        result.append(f"  Levenshtein Similarity: {overall['levenshtein_similarity']:.4f} (1.0 = identical)")
        result.append(f"  Edit Distance: {overall['edit_distance']} operations")
        result.append(f"  Reference Bigrams: {overall['input_count']}")
        result.append(f"  Hypothesis Bigrams: {overall['output_count']}")
        
        # Deterministic metrics
        det = metrics["deterministic"]
        result.append("\nDeterministic Bigrams:")
        result.append(f"  Levenshtein Similarity: {det['levenshtein_similarity']:.4f} (1.0 = identical)")
        result.append(f"  Edit Distance: {det['edit_distance']} operations")
        result.append(f"  Reference Bigrams: {det['input_count']}")
        result.append(f"  Hypothesis Bigrams: {det['output_count']}")
        
        # Non-deterministic metrics
        non_det = metrics["non_deterministic"]
        result.append("\nNon-Deterministic Bigrams:")
        result.append(f"  Levenshtein Similarity: {non_det['levenshtein_similarity']:.4f} (1.0 = identical)")
        result.append(f"  Edit Distance: {non_det['edit_distance']} operations")
        result.append(f"  Reference Bigrams: {non_det['input_count']}")
        result.append(f"  Hypothesis Bigrams: {non_det['output_count']}")
        
        return "\n".join(result)


# Simple usage example
def compare_sequences(reference: List[Any], hypothesis: List[Any], formatted: bool = True) -> Union[Dict, str]:
    """
    Wrapper function to easily compare two token sequences.
    
    Args:
        reference: Reference token sequence
        hypothesis: Hypothesis token sequence to compare
        formatted: Whether to return a formatted string (True) or dictionary (False)
        
    Returns:
        Comparison metrics as formatted string or dictionary
    """
    return BigramComparator.compare_token_sequences(reference, hypothesis, return_formatted=formatted)


if __name__ == "__main__":
    # Example usage
    reference = ['a', 'b', 'c', 'a', 'c', 'a', 'f', 'b', 'c']
    cases = [
        ['a', 'd', 'c', 'a', 'c', 'a', 'f', 'b', 'c'],  # a hallucinated token, instead of the correct one
        ['a', 'c', 'a', 'c', 'a', 'f', 'b', 'c'],       # a missed token - a is not followed by 'b' ; both deterministic & non-deterministic should get hurt ? -- currently yes ; because a,b & b,c ; change to only hurting in the case of the missing ab and not the missing bc ; because it didn't even show up ; so why should it hurt ? The determinism was never checked.
        ['a', 'b', 'c', 'a', 'c', 'f', 'b', 'c'],       # a missed 
        # ['a', 'b', 'c', 'a', 'f', 'b', 'c'],       # only a single missed start non-deterministic token; but that causes an avalanche ? 
        # ['a', 'b', 'c', 'g', 'c', 'a', 'f', 'b', 'c'],  # an additional hallucinated token (not replaced)
    ]
    
    print(f"Reference sequence: {reference}")
    
    for i, case in enumerate(cases):
        print(f"\n\n--- Case {i+1}: {case} ---")
        # Simple interface for comparing sequences
        result = compare_sequences(reference, case)
        print(result)