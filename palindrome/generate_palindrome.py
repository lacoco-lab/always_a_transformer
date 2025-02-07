import sys
from palindrome_generator import generate_strings, validate_palindrome, reverse_string

"""
This is a script to generate valid strings of unseen tokens to later make them into palindromes.

To generate strings, run:
python palindrome/generate_palindrome.py 10 100

where 10 is the number of tokens to sample and 100 is the number of their permutations in the string

Strings are saved as a numpy file in the datasets/flipflop.
"""

n, max_permutations = sys.argv[1:]
path = "datasets/palindrome"



