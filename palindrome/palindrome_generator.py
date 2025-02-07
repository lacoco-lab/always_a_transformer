import random
import math

def validate_palindrome(line):
    """
    Determine whether a string is a palindrome or not.
    
    :param line (str): The string to be checked
    :return: boolean, True or False
    """
    return line == line[::-1]


def reverse_string(line):
    """
    Make a string into a palindrome.
    
    :param line (str): The string to be reversed 
    :return: reversed string
    """
    
    return line[::-1]


def generate_strings(n, file="tokens.txt", max_permutations=None):
    """
    Generate exactly max_permutations random permutations of a list of tokens without generating all permutations.

    Args:
        tokens (list): List of tokens.
        max_permutations (int): Number of random permutations to generate.

    Returns:
        list: List of random permutations of the tokens.
    """
    # Generating all permutations (e.g. of 100 tokens) is extremely expensive, and we don't need that
    tokens = []
    with open(file, 'r') as file:
        for line in file:
            tokens.append(line.strip())

    assert len(tokens) >= n # Should have enough tokens to sample from

    tokens = random.sample(tokens, n) 
    
    assert max_permutations <= math.factorial(len(tokens)) # Make sure we are not asking for more than possible

    seen_permutations = set()
    result = []
    while len(result) < max_permutations:
        perm = tuple(random.sample(tokens, len(tokens)))
        if perm not in seen_permutations:
            seen_permutations.add(perm)
            result.append(''.join(perm))

    return result
