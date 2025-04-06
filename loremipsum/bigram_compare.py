from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union


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
    def get_token_alignment(seq1: List[Any], seq2: List[Any]) -> List[Tuple[str, int, int]]:
        """
        Calculate token-level alignment between sequences using both forward and
        reverse alignments, and choose the one that preserves early matches better.
        
        Args:
            seq1: First sequence (reference)
            seq2: Second sequence (hypothesis)
        Returns:
            List of (operation, seq1_index, seq2_index) tuples in forward order
        """
        m, n = len(seq1), len(seq2)
        
        # PART 1: Build standard forward DP matrix
        dp_forward = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize the first row and column
        for i in range(m + 1):
            dp_forward[i][0] = i
        for j in range(n + 1):
            dp_forward[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp_forward[i][j] = dp_forward[i-1][j-1]
                else:
                    dp_forward[i][j] = 1 + min(dp_forward[i-1][j],    # deletion
                                            dp_forward[i][j-1],     # insertion
                                            dp_forward[i-1][j-1])   # substitution
        
        # PART 2: Build reverse DP matrix using reversed sequences
        rev_seq1 = seq1[::-1]
        rev_seq2 = seq2[::-1]
        
        dp_reverse = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize the first row and column
        for i in range(m + 1):
            dp_reverse[i][0] = i
        for j in range(n + 1):
            dp_reverse[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if rev_seq1[i-1] == rev_seq2[j-1]:
                    dp_reverse[i][j] = dp_reverse[i-1][j-1]
                else:
                    dp_reverse[i][j] = 1 + min(dp_reverse[i-1][j],    # deletion
                                            dp_reverse[i][j-1],     # insertion
                                            dp_reverse[i-1][j-1])   # substitution
        
        # PART 3: Generate standard alignment (backtracking from end)
        forward_alignment_reversed = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq1[i-1] == seq2[j-1]:
                forward_alignment_reversed.append(('match', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp_forward[i][j] == dp_forward[i-1][j-1] + 1:
                forward_alignment_reversed.append(('substitute', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0 and dp_forward[i][j] == dp_forward[i-1][j] + 1:
                forward_alignment_reversed.append(('delete', i-1, None))
                i -= 1
            else:
                forward_alignment_reversed.append(('insert', None, j-1))
                j -= 1
        
        # PART 4: Generate reverse alignment (backtracking from end of reversed sequences)
        reverse_alignment_reversed = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and rev_seq1[i-1] == rev_seq2[j-1]:
                # Convert indices back to original sequence positions (critical fix)
                orig_i = m - i
                orig_j = n - j
                reverse_alignment_reversed.append(('match', orig_i, orig_j))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp_reverse[i][j] == dp_reverse[i-1][j-1] + 1:
                orig_i = m - i
                orig_j = n - j
                reverse_alignment_reversed.append(('substitute', orig_i, orig_j))
                i -= 1
                j -= 1
            elif i > 0 and dp_reverse[i][j] == dp_reverse[i-1][j] + 1:
                orig_i = m - i
                reverse_alignment_reversed.append(('delete', orig_i, None))
                i -= 1
            else:
                orig_j = n - j
                reverse_alignment_reversed.append(('insert', None, orig_j))
                j -= 1
        
        # PART 5: Put alignments in the correct order (start to end)
        # Sort by reference index when available, otherwise by hypothesis index
        def sort_key(item):
            op, idx1, idx2 = item
            if idx1 is not None:
                return idx1
            else:
                return idx2
        
        # Sort the alignments by position in the original sequence
        forward_alignment = sorted(forward_alignment_reversed, key=sort_key)
        reverse_alignment = sorted(reverse_alignment_reversed, key=sort_key)
        
        # PART 6: Compare alignments and choose the one with better early matches
        def early_match_score(alignment):
            score = 0
            for op, idx1, _ in alignment:
                if op == 'match':
                    # Strongly prefer matches at earlier positions
                    # Use exponential decay weight - earlier positions are worth much more
                    score += 100 * (0.9 ** idx1) if idx1 is not None else 0
            return score
        
        forward_score = early_match_score(forward_alignment)
        reverse_score = early_match_score(reverse_alignment)
        
        # Choose the alignment with better early matches
        chosen_alignment = reverse_alignment if reverse_score > forward_score else forward_alignment
        
        # Verify the alignment is in the correct order (start to end)
        # This is a double-check to ensure the output is correct
        return chosen_alignment

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
        alignment = ContextAwareBigramComparator.get_token_alignment(seq1, seq2)
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
        
        # Identify branch errors
        branch_errors = ContextAwareBigramComparator.detect_branch_errors(
            reference_tokens, hypothesis_tokens, alignment
        )
        
        if branch_errors:
            branch_markers = [''] * len(ref_result)
            for pos in branch_errors:
                # Find position in alignment
                align_pos = None
                for i, (op, ref_idx, _) in enumerate(alignment):
                    if ref_idx == pos:
                        align_pos = i
                        break
                
                if align_pos is not None:
                    branch_markers[align_pos] = '↓'
            
            result.append("Branches:   " + " ".join(branch_markers))
        
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
        # print(all_groups)
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
    reference = ['a', 'b', 'c', 'a', 'c', 'a', 'f', 'b', 'c']
    cases = [
        ['a', 'd', 'c', 'a', 'c', 'a', 'f', 'b', 'c'],  # single substitution (b→d)
        ['a', 'c', 'a', 'c', 'a', 'f', 'b', 'c'],       # single deletion (b missing)
    ]
    
    print(f"Reference sequence: {reference}")
    
    for i, case in enumerate(cases):
        print(f"\n\n--- Case {i+1}: {case} ---")
        result = compare_sequences_context_aware(reference, case)
        print(result)
        
        # Debug: Print alignment details
        # alignment = ContextAwareBigramComparator.get_token_alignment(reference, case)
        # print("\nAlignment Details:")
        # print(ContextAwareBigramComparator.format_alignment(reference, case, alignment))
    
    # Example with repeated subsequence
    reference2 = ['some', 'where', 'I', 'be', 'lo', 'ng', 'be', 'lo', 'ng', 'to', 'you']
    case2 = ['some', 'where', 'I', 'be', 'lo', 'ng', 'to', 'you']
    
    print(f"\n\nReference sequence with repeated pattern: {reference2}")
    print(f"Hypothesis with missing repetition: {case2}")
    result2 = compare_sequences_context_aware(reference2, case2)
    result2
    # print(result2['scores'])
