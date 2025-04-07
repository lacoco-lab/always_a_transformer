import argparse
import random
import pathlib
from pathlib import Path
from typing import List, Tuple, Set

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def test_character_tokenization(
        tokenizer,
        characters: List[str],
        with_space: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Test which characters tokenize as single tokens.

    Args:
        tokenizer: The tokenizer to test with
        characters: List of characters to test
        with_space: Whether to add spaces between characters

    Returns:
        Tuple of (good_chars, bad_chars) where:
            - good_chars are those that tokenize as single tokens
            - bad_chars are those that tokenize as multiple tokens
    """
    good_chars = []
    bad_chars = []

    for char in tqdm(characters, desc="Testing characters"):
        if with_space:
            tokens = tokenizer.tokenize(f" {char}", add_special_tokens=False)
        else:
            tokens = tokenizer.tokenize(char, add_special_tokens=False)

        # Check if the character tokenizes as a single token
        if len(tokens) == 1 and ((with_space and len(tokens[0]) <= 2) or
                                 (not with_space and len(tokens[0]) <= 1)):
            good_chars.append(char)
        else:
            bad_chars.append(char)

    return good_chars, bad_chars


def test_string_tokenization(
        tokenizer,
        test_string: str,
        with_space: bool = False
) -> bool:
    """
    Test if a string passes the tokenization check.

    Args:
        tokenizer: The tokenizer to test with
        test_string: The string to test
        with_space: Whether to add spaces between characters

    Returns:
        True if the string passes the check, False otherwise
    """
    if with_space:
        tokens = tokenizer.tokenize(" ".join(list(test_string)), add_special_tokens=False)
    else:
        tokens = tokenizer.tokenize(test_string, add_special_tokens=False)

    for token in tokens:
        if (with_space and len(token) > 2) or (not with_space and len(token) > 1):
            return False

    return True


def find_compatible_characters(
        tokenizer,
        num_needed: int = 2,
        with_space: bool = False,
        test_chars: Set[str] = None,
        excluded_chars: Set[str] = None
) -> List[str]:
    """
    Find characters that tokenize correctly and can replace digits.

    Args:
        tokenizer: The tokenizer to test with
        num_needed: Number of characters needed (default: 2)
        with_space: Whether to add spaces between characters
        test_chars: Set of characters to test (if None, uses a default set)
        excluded_chars: Set of characters to exclude from testing

    Returns:
        List of compatible characters
    """
    if excluded_chars is None:
        excluded_chars = set()

    if test_chars is None:
        # Create a range of possible replacement characters
        # Excluding digits, punctuation, and the characters already used in the pattern ('w', 'r', 'i')
        test_chars = set()

        # Add lowercase letters (excluding w, r, i)
        test_chars.update([chr(c) for c in range(ord('a'), ord('z') + 1)
                           if chr(c) not in {'w', 'r', 'i'}])

        # Add uppercase letters
        test_chars.update([chr(c) for c in range(ord('A'), ord('Z') + 1)])
        
        # Add other digits
        test_chars.update([chr(c) for c in range(ord('0'), ord('9') + 1) if chr(c) not in {'0', '1'}])

    # Remove any excluded characters
    test_chars = {c for c in test_chars if c not in excluded_chars}

    # If no characters left to test, return empty list
    if not test_chars:
        return []

    good_chars, _ = test_character_tokenization(tokenizer, list(test_chars), with_space)

    # If we don't have enough good characters, we'll need to expand our search
    if len(good_chars) < num_needed:
        print(f"Warning: Only found {len(good_chars)} compatible characters, need {num_needed}")
        return good_chars[:num_needed]  # Return whatever we have

    return good_chars[:num_needed]


def generate_string(length: int, digit_replacements: List[str]) -> str:
    """
    Generate a string following these rules:
    1. Contains only 'w', 'r', 'i', and characters from digit_replacements
    2. Starts with 'r' or 'i' and ends with a replacement digit (always even length)
    3. Contains exactly one 'w', not at the first position
    4. Replacement digits occur randomly at odd positions

    Args:
        length: Desired length of the string (adjusted to be even and at least 4)
        digit_replacements: Characters to use in place of digits

    Returns:
        A string following the specified rules
    """
    # Ensure length is even and at least 4 (minimum needed for requirements)
    if length < 4:
        length = 4
    elif length % 2 != 0:
        length += 1

    # Initialize the string
    result = [''] * length

    # First character must be 'r' or 'i' (not 'w')
    result[0] = random.choice(['r', 'i'])

    # Choose a random non-first even position for 'w'
    # Even positions are 0, 2, 4, etc. (we need to exclude position 0)
    w_position = random.choice(range(2, length, 2))
    result[w_position] = 'w'

    # Fill in the remaining characters at even positions
    for i in range(2, length, 2):
        if i != w_position:  # Skip the position where 'w' is placed
            result[i] = random.choice(['r', 'i'])

    # Fill in the replacement digits at odd positions
    for i in range(1, length, 2):
        result[i] = random.choice(digit_replacements)

    return ''.join(result)


def generate_samples(
        min_length: int,
        max_length: int,
        num_samples: int,
        digit_replacements: List[str]
) -> List[Tuple[int, int, str]]:
    """
    Generate multiple samples with a good representation of different lengths.

    Args:
        min_length: Minimum length of the strings (adjusted to be even and at least 4)
        max_length: Maximum length of the strings (adjusted to be even and at least min_length)
        num_samples: Number of samples to generate
        digit_replacements: Characters to use in place of digits

    Returns:
        List of tuples (sample_number, length, generated_string)
    """
    # Ensure min_length is even and at least 4
    if min_length < 4:
        min_length = 4
    elif min_length % 2 != 0:
        min_length += 1

    # Ensure max_length is even and at least min_length
    if max_length < min_length:
        max_length = min_length
    elif max_length % 2 != 0:
        max_length += 1

    # Generate possible lengths (all even numbers from min_length to max_length)
    possible_lengths = list(range(min_length, max_length + 1, 2))

    # Initialize a counter for each possible length
    length_counts = {length: 0 for length in possible_lengths}

    # Generate the samples
    samples = []
    for i in range(1, num_samples + 1):
        # Find the lengths with the minimum count so far
        min_count = min(length_counts.values())
        candidates = [length for length, count in length_counts.items() if count == min_count]

        # Randomly select one of the candidate lengths
        chosen_length = random.choice(candidates)

        # Generate a string of the chosen length
        result = generate_string(chosen_length, digit_replacements)

        # Update the count for this length
        length_counts[chosen_length] += 1

        # Store the sample info
        samples.append((i, chosen_length, result))

    return samples


def verify_samples(
        tokenizer,
        samples: List[Tuple[int, int, str]],
        with_space: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Verify that the generated samples pass the tokenization check.

    Args:
        tokenizer: The tokenizer to use
        samples: List of sample tuples (id, length, string)
        with_space: Whether to add spaces between characters

    Returns:
        Tuple of (passed_samples, failed_samples)
    """
    passed = []
    failed = []

    for _, _, sample in tqdm(samples, desc="Verifying samples"):
        if test_string_tokenization(tokenizer, sample, with_space):
            passed.append(sample)
        else:
            failed.append(sample)

    return passed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate tokenization-friendly strings')
    parser.add_argument('-min', '--min-length', type=int, default=4,
                        help='Minimum desired length of the strings (default: 4)')
    parser.add_argument('-max', '--max-length', type=int, required=True,
                        help='Maximum desired length of the strings')
    parser.add_argument('-n', '--samples', type=int, default=1,
                        help='Number of samples to generate (default: 1)')
    parser.add_argument('-o', '--outfile', type=str,
                        help='Output file path (default: prints to console only)')
    parser.add_argument('--tokenizer-name', type=str, required=True,
                        help='Tokenizer name (for compatibility testing)')
    parser.add_argument('--with-space', action='store_true',
                        help='Check for space in tokens')
    parser.add_argument('--specific-chars', type=str, default=None,
                        help='Specific characters to test (comma separated)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test characters without generating samples')
    parser.add_argument('--max-retries', type=int, default=10,
                        help='Maximum number of character set retries (default: 10)')
    args = parser.parse_args()

    try:
        print(f"Loading tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        # Define specific characters to test if provided
        initial_test_chars = None
        if args.specific_chars:
            initial_test_chars = set(args.specific_chars.split(','))
            print(f"Starting with specific characters: {initial_test_chars}")

        if args.test_only:
            # Just find some compatible characters without generating
            compatible_chars = find_compatible_characters(
                tokenizer,
                num_needed=2,
                with_space=args.with_space,
                test_chars=initial_test_chars
            )
            print(f"Found compatible characters: {compatible_chars}")
            print("Test-only mode. Exiting without generating samples.")
            exit(0)

        # Iterative process to find characters that work for all samples
        excluded_chars = set()  # Characters we've already tried
        retry_count = 0
        all_passed = False
        final_samples = []

        while not all_passed and retry_count < args.max_retries:
            # Find compatible characters, excluding ones we've already tried
            print(f"\nTry #{retry_count + 1}: Finding compatible characters...")
            compatible_chars = find_compatible_characters(
                tokenizer,
                num_needed=2,
                with_space=args.with_space,
                test_chars=initial_test_chars,
                excluded_chars=excluded_chars
            )

            if not compatible_chars or len(compatible_chars) < 2:
                print("Couldn't find enough compatible characters. Exiting.")
                break

            print(f"Found compatible characters: {compatible_chars}")

            # Add these characters to our excluded set for potential future retries
            excluded_chars.update(compatible_chars)

            # Generate samples using compatible characters
            print(f"Generating {args.samples} samples with replacement characters: {compatible_chars}")
            samples = generate_samples(
                args.min_length,
                args.max_length,
                args.samples,
                compatible_chars
            )

            # Verify samples
            passed_samples, failed_samples = verify_samples(
                tokenizer,
                samples,
                args.with_space
            )

            pass_rate = len(passed_samples) / len(samples) * 100 if samples else 0
            print(f"Generated {len(samples)} samples")
            print(f"Passed tokenization check: {len(passed_samples)} ({pass_rate:.2f}%)")
            print(f"Failed tokenization check: {len(failed_samples)}")

            if not failed_samples:
                print("All samples passed! Success!")
                all_passed = True
                final_samples = passed_samples
            else:
                print(f"Some samples failed. Retrying with different characters...")
                if failed_samples:
                    print("Example of failing sample:")
                    print(failed_samples[0])
                retry_count += 1

        if not all_passed:
            print(f"\nFailed to find characters that work for all samples after {args.max_retries} attempts.")
            print("Using the best result we found so far.")
            final_samples = passed_samples

        # Print sample examples of our final result
        if final_samples:
            print("\nExample of final passing sample:")
            print(final_samples[0])

            # Write final samples to file if specified
            if args.outfile:
                # Convert to Path object
                outfile_path = pathlib.Path(args.outfile)

                # Create parent directory if it doesn't exist
                outfile_path.parent.mkdir(parents=True, exist_ok=True)

                # Write to file
                with open(outfile_path, 'w') as f:
                    for sample in final_samples:
                        f.write(f"{sample}\n")

                print(f"\nFinal samples written to: {outfile_path}")
        else:
            print("No passing samples found. No output file created.")

    except ValueError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error writing to file: {e}")