import random
import argparse
import pathlib
import json
import string
from collections import Counter
from tqdm.auto import tqdm


def generate_string(length):
    """
    Generate a random string with letters and digits.

    Args:
        length (int): Desired length of the string

    Returns:
        str: A random string with letters and digits
    """
    # Generate random string with letters and digits
    letters = string.ascii_lowercase  # a through z
    digits = string.digits  # 0 through 9

    # Start with a random sequence of characters
    result = []
    for _ in range(length):
        # Decide if this position will be a letter or digit (50/50 chance)
        if random.random() < 0.5:
            result.append(random.choice(letters))
        else:
            result.append(random.choice(digits))

    return ''.join(result)


def find_adjacent_characters(string, position):
    """
    Find the immediately adjacent characters to the left and right of a given position.

    Args:
        string (str): The input string
        position (int): The position of the character

    Returns:
        tuple: (left_char, right_char) where each may be None if at an edge
    """
    left_char = string[position - 1] if position > 0 else None
    right_char = string[position + 1] if position < len(string) - 1 else None

    return left_char, right_char


def generate_valid_sample(min_length, max_length, answer_type):
    """
    Generate a valid sample where a unique character is chosen.

    Args:
        min_length (int): Minimum length of the strings
        max_length (int): Maximum length of the strings
        answer_type (str): Type of answer to generate ('left' or 'right')

    Returns:
        dict: Dictionary containing 'input' and 'golden_answer', or None if no valid sample could be created
    """
    # Choose a random length
    length = random.randint(min_length, max_length)

    # Generate a string
    string_value = generate_string(length)

    # Count occurrences of each character
    char_counter = Counter(string_value)

    # Find unique characters (appearing exactly once) that are letters or digits
    unique_chars = [char for char, count in char_counter.items()
                    if count == 1]  # Allow both letters and digits

    # If we have unique characters
    if unique_chars:
        # Choose a random unique character
        chosen_char = random.choice(unique_chars)

        # Find position of the chosen character
        char_position = string_value.index(chosen_char)

        # Find adjacent characters
        left_char, right_char = find_adjacent_characters(string_value, char_position)

        # Determine the answer based on configuration
        if answer_type == 'left':
            answer = left_char if left_char is not None else '-1'
        else:  # answer_type == 'right'
            answer = right_char if right_char is not None else '-1'

        # Create the input string with separator and unique character
        input_string = string_value + '||' + chosen_char

        return {
            'input': input_string,
            'golden_answer': answer
        }

    return None  # No unique characters found


def generate_invalid_sample(min_length, max_length):
    """
    Generate an invalid sample where the character after the separator doesn't appear in the string.

    Args:
        min_length (int): Minimum length of the strings
        max_length (int): Maximum length of the strings

    Returns:
        dict: Dictionary containing 'input' and 'golden_answer'
    """
    # Choose a random length
    length = random.randint(min_length, max_length)

    # Generate a string
    string_value = generate_string(length)

    # Find characters that don't appear in the string
    all_chars = set(string.ascii_lowercase + string.digits)  # Include digits as potential queries
    used_chars = set(string_value)
    unused_chars = all_chars - used_chars

    # If all possible characters are used (very unlikely for longer strings),
    # we'll modify the string to free up some characters
    if not unused_chars:
        # Remove one character type by replacing it
        char_to_replace = random.choice(list(used_chars))
        replacement = random.choice(list(used_chars - {char_to_replace}))
        string_value = string_value.replace(char_to_replace, replacement)

        # Recalculate unused characters
        used_chars = set(c for c in string_value if c.isalpha())
        unused_chars = all_chars - used_chars

    # Choose a character that doesn't appear in the string
    chosen_char = random.choice(list(unused_chars))

    # Create the input string with separator and chosen character
    input_string = string_value + '||' + chosen_char

    return {
        'input': input_string,
        'golden_answer': '-1'
    }


def generate_samples(min_length, max_length, num_samples, answer_type='right'):
    """
    Generate multiple samples with JSON structure, ensuring no duplicates.

    Args:
        min_length (int): Minimum length of the strings
        max_length (int): Maximum length of the strings
        num_samples (int): Number of samples to generate
        answer_type (str): Type of answer to generate ('left' or 'right')

    Returns:
        list: List of dictionaries containing 'input' and 'golden_answer'
    """
    samples = []
    unique_inputs = set()  # Track unique input strings to avoid duplicates

    # Calculate how many invalid samples we need (10% of total)
    num_invalid = max(1, int(num_samples * 0.1))
    num_valid = num_samples - num_invalid

    # Generate valid samples
    valid_count = 0
    max_attempts = num_valid * 100  # Increased to handle potential collisions
    attempts = 0

    # Create progress bar for valid samples
    valid_pbar = tqdm(total=num_valid, desc="Generating valid samples", unit="sample")

    while valid_count < num_valid and attempts < max_attempts:
        sample = generate_valid_sample(min_length, max_length, answer_type)
        if sample and sample['input'] not in unique_inputs:
            samples.append(sample)
            unique_inputs.add(sample['input'])
            valid_count += 1
            valid_pbar.update(1)
        attempts += 1

        # If we're struggling to find unique valid samples after many attempts
        if attempts > max_attempts * 0.8 and valid_count < num_valid * 0.8:
            # Try increasing the length range slightly to get more variety
            max_length += 1

    valid_pbar.close()

    # Generate invalid samples
    invalid_count = 0
    max_invalid_attempts = num_invalid * 5000  # Much higher to ensure we find enough unique invalid samples
    invalid_attempts = 0

    # Create progress bar for invalid samples
    invalid_pbar = tqdm(total=num_invalid, desc="Generating invalid samples", unit="sample")

    while invalid_count < num_invalid and invalid_attempts < max_invalid_attempts:
        sample = generate_invalid_sample(min_length, max_length)
        if sample['input'] not in unique_inputs:
            samples.append(sample)
            unique_inputs.add(sample['input'])
            invalid_count += 1
            invalid_pbar.update(1)
        invalid_attempts += 1

        # If we're struggling to find unique invalid samples
        if invalid_attempts > max_invalid_attempts * 0.8 and invalid_count < num_invalid * 0.8:
            # Try increasing the length range slightly
            max_length += 1

    invalid_pbar.close()

    # Print warning if we couldn't generate the requested number of samples
    if len(samples) < num_samples:
        print(f"Warning: Could only generate {len(samples)} unique samples "
              f"out of {num_samples} requested.")

    # Shuffle the samples
    random.shuffle(samples)

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate strings following generic rules')
    parser.add_argument('-min', '--min-length', type=int, default=4,
                        help='Minimum desired length of the strings (default: 4)')
    parser.add_argument('-max', '--max-length', type=int, required=True,
                        help='Maximum desired length of the strings')
    parser.add_argument('-n', '--samples', type=int, default=1,
                        help='Number of samples to generate (default: 1)')
    parser.add_argument('-a', '--answer-type', type=str, default='right', choices=['left', 'right'],
                        help='Type of answer to generate (default: right)')
    parser.add_argument('-o', '--outfile', type=str,
                        help='Output file path (default: prints to console only)')
    parser.add_argument('-s', '--seed', type=int, default=51,)
    args = parser.parse_args()
    
    # Set the random seed for reproducibility
    random.seed(args.seed)

    try:
        # Generate the samples
        samples = generate_samples(args.min_length, args.max_length, args.samples, args.answer_type)

        print(
            f"Generated {len(samples)} unique samples ({len([s for s in samples if s['golden_answer'] != '-1'])} valid, "
            f"{len([s for s in samples if s['golden_answer'] == '-1'])} invalid)")

        # Write the data as jsonlines (each line is a separate JSON object)
        if args.outfile:
            # Convert to Path object
            outfile_path = pathlib.Path(args.outfile)

            # Create parent directory if it doesn't exist
            outfile_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file as jsonlines
            with open(outfile_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')

            print(f"\nSamples written to: {outfile_path}")
        else:
            # Print each sample as a separate JSON object
            for sample in samples:
                print(json.dumps(sample))

    except ValueError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error writing to file: {e}")