import sys
import numpy as np
from flipflop_generator import generate_all_valid_flipflops, validate_flip_flop


"""
This is a script to generate valid FlipFlop strings in the needed range.
Provide ranges as command line arguments by running:

python flipflop/generate_flipflop.py 4 20

where 4 is the lower boundary and 20 is the upper boundary.

Strings are saved as a numpy file in the datasets/flipflop.
"""


start_length, finish_length = sys.argv[1:]
all_valid_flipflops = []
path = "datasets/flipflop"

for length in range(int(start_length), int(finish_length)+1, 2):
    valid_flipflops = generate_all_valid_flipflops(int(length))

    # Validate all generated strings and only keep the valid ones
    valid_flipflops_count = 0
    for flipflop in valid_flipflops:
        try:
            validate_flip_flop(flipflop)
            all_valid_flipflops.append(flipflop)
            valid_flipflops_count += 1
        except AssertionError:
            continue

    print(f"Generated {valid_flipflops_count} strings for length {length}.")

np.savetxt(path + f"/flipflop_{finish_length}.txt", all_valid_flipflops, delimiter='\n', fmt='%s')

print(f"Saved {len(all_valid_flipflops)} valid FlipFlop strings to {path}/flipflop_{finish_length}.txt.")
