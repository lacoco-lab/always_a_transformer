from itertools import product
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Any, Union


def needleman_wunsch(x, y):
    """Run the Needleman-Wunsch algorithm on two sequences.

    x, y -- sequences.

    Code based on pseudocode in Section 3 of:

    Naveed, Tahir; Siddiqui, Imitaz Saeed; Ahmed, Shaftab.
    "Parallel Needleman-Wunsch Algorithm for Grid." n.d.
    https://upload.wikimedia.org/wikipedia/en/c/c4/ParallelNeedlemanAlgorithm.pdf
    """
    N, M = len(x), len(y)
    s = lambda a, b: int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    return list(alignment)


def align_fast(x, y):
    """Align two sequences, maximizing the
    alignment score, using the Needleman-Wunsch
    algorithm.

    x, y -- sequences.
    """
    alignment = needleman_wunsch(x, y)
    new_alignment = []
    for x_idx, y_idx in alignment:
        if x_idx is None and y_idx is not None:
            new_alignment.append(('insert', x_idx, y_idx))
        elif y_idx is None:
            new_alignment.append(('delete', x_idx, y_idx))
        elif (x[x_idx] == y[y_idx]):
            new_alignment.append(('match', x_idx, y_idx))            
        elif x_idx is not None and y_idx is not None:
            new_alignment.append(('substitute', x_idx, y_idx))
    return new_alignment


def print_alignment(x, y, alignment):
    print("".join(
        "-" if i is None else x[i] for i, _ in alignment
    ))
    print("".join(
        "-" if j is None else y[j] for _, j in alignment
    ))


class ContextAwareBigramComparator:
    """
    A utility class for comparing token sequences using context-aware bigram analysis.
    This implementation counts consecutive errors as a single logical error group,
    avoiding over-penalization for cascading mistakes.
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
        
        return deterministic, non_deterministic

    @staticmethod
    def group_operations(alignment: List[Tuple[str, Any, Any]]) -> List[Tuple[str, List[Tuple[Any, Any]]]]:
        """
        Group consecutive operations of the same type.
        
        Args:
            alignment: List of (operation, seq1_index, seq2_index) tuples
            
        Returns:
            List of (operation, [(seq1_index, seq2_index), ...]) tuples
        """
        if not alignment:
            return []
        
        grouped = []
        current_op = alignment[0][0]
        current_indices = [(alignment[0][1], alignment[0][2])]
        
        for op, idx1, idx2 in alignment[1:]:
            if op == current_op:
                # Continue current group
                current_indices.append((idx1, idx2))
            else:
                # End current group and start a new one
                grouped.append((current_op, current_indices))
                current_op = op
                current_indices = [(idx1, idx2)]
        
        # Add the last group
        grouped.append((current_op, current_indices))
        return grouped
    
    @staticmethod
    def context_aware_distance(seq1: List[Any], seq2: List[Any]) -> int:
        """
        Calculate a context-aware distance that counts groups of operations.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Number of edit operation groups
        """
        # TO DO : Need to handle empty sequences
        # Get the alignment
        alignment = align_fast(seq1, seq2)
        print("Alignment:", alignment)
        # Group operations
        return ContextAwareBigramComparator.group_operations(alignment)
        
    
    @staticmethod
    def format_alignment(reference_tokens: List[Any], hypothesis_tokens: List[Any], alignment: List[Tuple[str, Any, Any]]) -> str:
        """Debug utility to visualize the alignment between sequences."""
        ref_result = []
        hyp_result = []
        op_result = []
        
        print(alignment)
        for op, ref_idx, hyp_idx in alignment:
            if op == 'match':
                ref_result.append(str(reference_tokens[ref_idx]))
                hyp_result.append(str(hypothesis_tokens[hyp_idx]))
                op_result.append('=')
            elif op == 'substitute':
                ref_result.append(str(reference_tokens[ref_idx]))
                hyp_result.append(str(hypothesis_tokens[hyp_idx]))
                op_result.append('≠')
            elif op == 'delete':
                ref_result.append(str(reference_tokens[ref_idx]))
                hyp_result.append('-')
                op_result.append('D')
            elif op == 'insert':
                ref_result.append('-')
                hyp_result.append(str(hypothesis_tokens[hyp_idx]))
                op_result.append('I')
        
        result = []
        result.append("Reference:  " + " ".join(ref_result))
        result.append("Hypothesis: " + " ".join(hyp_result))
        result.append("Operations: " + " ".join(op_result))        
        return "\n".join(result)
    
    @staticmethod
    def classify_alignments(grouped_alignment, 
                        deterministic_tokens, 
                        non_deterministic_tokens,
                        reference_tokens):
        """
        Classify edits as deterministic or non-deterministic based on transition points.
        
        Args:
            grouped_alignment: List of (operation, [(ref_idx, hyp_idx), ...]) tuples
            deterministic_tokens: Set of tokens that are considered deterministic
            non_deterministic_tokens: Set of tokens that are considered non-deterministic
            reference_tokens: Original reference sequence of tokens
            
        Returns:
            Dict with classification results and scores
        """
        result = {
            "edit_classifications": [],
            "statistics": {
                "deterministic_edits": 0,
                "non_deterministic_edits": 0,
                "unknown_edits": 0,
                "total_edits": 0
            }
        }
        
        # print("deterministic", deterministic_tokens)
        # print("non deterministic", non_deterministic_tokens)
        print(grouped_alignment)
        # Process each transition between match and edit operations
        for i in range(len(grouped_alignment) - 1):
            current_op, current_indices = grouped_alignment[i]
            next_op, next_indices = grouped_alignment[i+1]
            
            # We're only interested in transitions from match to edit operations
            if current_op == "match" and next_op in ["substitute", "delete", "insert"]:
                # Get the last token from the match group
                last_match_ref_idx = current_indices[-1][0]
                last_token = reference_tokens[last_match_ref_idx]
                
                # Check if the token is in our deterministic/non-deterministic lists
                is_deterministic = last_token in deterministic_tokens
                is_non_deterministic = last_token in non_deterministic_tokens
                
                result["statistics"]["total_edits"] += 1
                
                # Classify the edit
                if is_deterministic:
                    classification = "deterministic"
                    result["statistics"]["deterministic_edits"] += 1
                elif is_non_deterministic:
                    classification = "non_deterministic"
                    result["statistics"]["non_deterministic_edits"] += 1
                else:
                    classification = "non_deterministic"
                    result["statistics"]["non_deterministic_edits"] += 1
                
                # Add details for this edit group
                edit_info = {
                    "transition_point": last_match_ref_idx,
                    "transition_token": last_token,
                    "edit_operation": next_op,
                    "edit_indices": next_indices,
                    "classification": classification
                }
                
                # Add the specific tokens being edited if available
                edited_tokens = []
                if next_op == "delete" or next_op == "substitute":
                    for idx, _ in next_indices:
                        edited_tokens.append(reference_tokens[idx])
                
                edit_info["edited_tokens"] = edited_tokens
                result["edit_classifications"].append(edit_info)
        
        # Also check transitions from edit to match (for completeness)
        for i in range(len(grouped_alignment) - 1):
            current_op, current_indices = grouped_alignment[i]
            next_op, next_indices = grouped_alignment[i+1]
            
            if current_op in ["substitute", "delete", "insert"] and next_op == "match":
                # For transitions to match, we check the first token of the match
                first_match_ref_idx = next_indices[0][0]
                first_token = reference_tokens[first_match_ref_idx]
                
                # Add this transition point to results (without affecting counts)
                result["edit_classifications"].append({
                    "transition_point": first_match_ref_idx,
                    "transition_token": first_token,
                    "edit_operation": "transition_to_match",
                    "classification": "transition_only"  # Not counted in totals
                })
        
        # Calculate scores
        total = len(reference_tokens)
        det_total = len(deterministic_tokens)
        non_det_total = len(deterministic_tokens)
        if total > 0:
            result["scores"] = {
                "deterministic": 1 - (result["statistics"]["deterministic_edits"] / det_total),
                "non_deterministic": 1 - (result["statistics"]["non_deterministic_edits"] / non_det_total),
                "overall": 1 - (result["statistics"]["total_edits"] / total)
            }
        else:
            result["scores"] = {
                "deterministic_score": 0.0,
                "non_deterministic_score": 0.0
            }
        
        return result

    
    @classmethod
    def compare_token_sequences(cls, 
                               reference_tokens: List[Any], 
                               hypothesis_tokens: List[Any],
                               return_formatted: bool = False) -> Union[Dict, str]:
        """
        Compare two token sequences using context-aware bigram analysis.
        Each edit group is counted exactly once in the category where it originated.
        
        Args:
            reference_tokens: Reference sequence of tokens
            hypothesis_tokens: Hypothesis sequence of tokens to compare against reference
            return_formatted: If True, returns a formatted string instead of metrics dictionary
            
        Returns:
            Dictionary with detailed metrics or formatted string if return_formatted=True
        """
        # Get bigram versions
        ref_bigrams = cls.get_bigram_version(reference_tokens)        
        # Identify deterministic and non-deterministic tokens in reference
        det_firsts, non_det_firsts = cls.categorize_bigrams(ref_bigrams)
                
        # Get token-level edit groups for overall metric
        all_groups = cls.context_aware_distance(reference_tokens, hypothesis_tokens)        
        print(all_groups)
        result = cls.classify_alignments(all_groups, det_firsts, non_det_firsts, reference_tokens)
        return result


# Simple usage example
def compare_sequences_context_aware(reference: List[Any], hypothesis: List[Any], formatted: bool = True) -> Union[Dict, str]:
    """
    Wrapper function to easily compare two token sequences using context-aware approach.
    
    Args:
        reference: Reference token sequence
        hypothesis: Hypothesis token sequence to compare
        formatted: Whether to return a formatted string (True) or dictionary (False)
        
    Returns:
        Comparison metrics as formatted string or dictionary
    """
    return ContextAwareBigramComparator.compare_token_sequences(reference, hypothesis, return_formatted=formatted)


if __name__ == "__main__":
    # Example usage
    cases = [
        # (['a', 'b', 'c', 'a', 'c', 'a', 'f', 'b', 'c'], ['a', 'd', 'c', 'a', 'c', 'a', 'f', 'b', 'c']),  # single substitution (b→d)
        # (['a', 'b', 'c', 'a', 'c', 'a', 'f', 'b', 'c'], ['a', 'c', 'a', 'c', 'a', 'f', 'b', 'c']),       # single deletion (b missing)
        # (['Ip', 'sum', 'no', 'quem', 'do', '', 'lore', 's'], ['Ip', 'sum', 'no', 'be', 'lo', 'ng', 'to', 'you']), # a lot of consecutive deletions,
        (['Ġne', 'am', 'que', '.', 'ĠNon', 'Ġet', 'inc', 'idunt', 'Ġdol', 'orem', 'Ġtemp', 'ora', 'Ġmagn', 'am', '.'], ['Ġne', 'am', 'que', '.', 'ĠNon', 'Ġet', 'inc', 'idunt', 'Ġdol', 'orem', 'Ġtemp', 'ora', 'Ġmagn', 'am', 'Ġvelit', 'Ġne', 'que', '.', 'ĠNon', 'Ġet', 'inc', 'idunt', 'Ġdol', 'orem', 'Ġtemp', 'ora', 'Ġmagn', 'am', '.']) # Actual sample
    ]
    
    
    for idx, (reference, case) in enumerate(cases):
        print(f"\n\n--- Case {idx+1}: {case} ---")
        print(f"Reference sequence: {reference}")
        result = compare_sequences_context_aware(reference, case)
        print(result)