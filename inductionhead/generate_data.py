import random
import argparse
import pathlib


def generate_string(length):
    """
    Generate a string following these rules:
    1. Contains only 'w', 'r', 'i', '0', '1'
    2. Starts with 'r' or 'i' and ends with a digit (always even length)
    3. Contains exactly one 'w', not at the first position
    4. Digits (0, 1) occur randomly

    Args:
        length (int): Desired length of the string (adjusted to be even and at least 4)

    Returns:
        str: A string following the specified rules
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

    # Fill in the digits at odd positions
    for i in range(1, length, 2):
        result[i] = random.choice(['0', '1'])

    return ''.join(result)


def generate_samples(min_length, max_length, num_samples):
    """
    Generate multiple samples with a good representation of different lengths.

    Args:
        min_length (int): Minimum length of the strings (adjusted to be even and at least 4)
        max_length (int): Maximum length of the strings (adjusted to be even and at least min_length)
        num_samples (int): Number of samples to generate

    Returns:
        list: List of tuples (sample_number, length, generated_string)
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
        result = generate_string(chosen_length)

        # Update the count for this length
        length_counts[chosen_length] += 1

        # Store the sample info
        samples.append((i, chosen_length, result))

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate strings following specific rules')
    parser.add_argument('-min', '--min-length', type=int, default=4,
                        help='Minimum desired length of the strings (default: 4)')
    parser.add_argument('-max', '--max-length', type=int, required=True,
                        help='Maximum desired length of the strings')
    parser.add_argument('-n', '--samples', type=int, default=1,
                        help='Number of samples to generate (default: 1)')
    parser.add_argument('-o', '--outfile', type=str,
                        help='Output file path (default: prints to console only)')
    args = parser.parse_args()

    try:
        # Generate the samples
        samples = generate_samples(args.min_length, args.max_length, args.samples)

        # # Print samples to console
        # for i, length, result in samples:
        #     print(f"Sample {i} (length {length}): {result}")

        # Write samples to file if specified
        if args.outfile:
            # Convert to Path object
            outfile_path = pathlib.Path(args.outfile)

            # Create parent directory if it doesn't exist
            outfile_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(outfile_path, 'w') as f:
                for _, _, result in samples:
                    f.write(f"{result}\n")

            print(f"\nSamples written to: {outfile_path}")

    except ValueError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error writing to file: {e}")