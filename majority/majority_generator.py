import sys
from generate_majority import generate_majority

"""
This is a script to generate valid Majority strings in the needed range.
Provide ranges as command line arguments by running:

python flipflop/generate_flipflop.py min_length max_length step min_perc_ones max_perc_ones num_strings

Strings are saved as a txt file in the datasets/majority directory.
"""

min_length, max_length, step, min_ones, max_ones, num_strings = sys.argv[1:]

for i in range(min_length, max_length, step):
    generate_majority(
        n_strings=num_strings,
        length=i,
        min_percentage=min_ones,
        max_percentage=max_ones,
        output_dir="../datasets/majority/s1"
    )