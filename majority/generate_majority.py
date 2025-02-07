import random
import math


def analyze_majority(s: str):
    """
    Returns a tuple containing:
      1) length of the string,
      2) number of '1's in the string,
      3) percentage of '1's (0.0 if the string is empty).
      
    :param s: Majority string (string)
    """
    length = len(s)
    count_ones = s.count('1')
    assert length > 0
    
    percentage_ones = 100.0 * count_ones / length
    
    return length, count_ones, percentage_ones


def generate_majority(
        n_strings=100,
        length=20,
        min_percentage=1,
        max_percentage=10,
        max_attempts=None,
        output_dir='../datasets/majority/s1'
):
    """
    Generate up to `n_strings` unique binary strings of length `length`.
    Each string has between `min_percentage` and `max_percentage` percent 1s (inclusive).

    :param n_strings:     Number of unique binary strings to generate.
    :param length:        Length of each binary string.
    :param min_percentage: Minimum percentage of 1s in each string (integer).
    :param max_percentage: Maximum percentage of 1s in each string (integer).
    :param output_dir:      Output directory for the .txt files.
    :param max_attempts:  Maximum number of random attempts. If None, defaults to 10 * n_strings.
    """

    if max_attempts is None:
        max_attempts = 10 * n_strings

    min_ones = math.ceil(length * min_percentage / 100.0)
    max_ones = math.floor(length * max_percentage / 100.0)

    if min_ones < 0:
        raise ValueError("Minimum number of 1s cannot be negative.")
    if max_ones > length:
        raise ValueError("Maximum number of 1s cannot exceed the string length.")
    if min_ones > max_ones:
        raise ValueError("Invalid percentage range: no overlap in [min_ones, max_ones].")

    unique_strings = set()
    attempts = 0

    while len(unique_strings) < n_strings and attempts < max_attempts:
        k_ones = random.randint(min_ones, max_ones)
        positions = random.sample(range(length), k_ones)

        s = ["0"] * length
        for pos in positions:
            s[pos] = "1"
        s_str = "".join(s)

        unique_strings.add(s_str)
        attempts += 1

    if len(unique_strings) < n_strings:
        print(f"Warning: only generated {len(unique_strings)} unique strings "
              f"(less than requested {n_strings}).")
    
    output_file = output_dir + f"/b{min_percentage}-{max_percentage}/majority_{length}.txt"
    
    with open(output_file, "w") as f:
        for s_str in unique_strings:
            f.write(s_str + "\n")

    print(f"Successfully wrote {len(unique_strings)} strings of length {length} for the bin {min_percentage}-{max_percentage} to {output_file}.")


def is_binary_string(s: str) -> bool:
    """
    Make sure all characters in `s` are 0s or 1s.
    :param s: Majority string (string)
    :return: True if all characters in `s` are 0s or 1s.
    """
    return all(ch in ('0', '1') for ch in s)


def validate_majority(s, length, min_percentage, max_percentage):
    """
    Validate a majority string of length `length` with at least `min_percentage` of 1s and at max `max_percentage` of 1s.
    :param s: Majority string (string)
    :param min_percentage: Minimum percentage of 1s in a string (integer).
    :param max_percentage: Maximum percentage of 1s in a string (integer).
    :param length: Required length of each binary string (integer).
    :return: Boolean indicating whether `s` is valid.
    """

    if not is_binary_string(s):
        return False, "Non-binary characters found."
    
    if len(s) != length:
        return False, f"Expected length {length}, but got {len(s)}."

    count_ones = s.count('1')
    percentage_ones = (count_ones / length) * 100
    if percentage_ones < min_percentage or percentage_ones > max_percentage:
        return False, (f"Percentage of 1s out of range "
                       f"({percentage_ones:.2f}% not in [{min_percentage}, {max_percentage}]).")

    return True, f"Valid majority string of length {length} in bin {min_percentage}-{max_percentage}."
