import sys
from itertools import chain
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

    # Validate all generated strings
    for flipflop in valid_flipflops:
        validate_flip_flop(flipflop)

    all_valid_flipflops.append(valid_flipflops)

    print(f"Generated {len(valid_flipflops)} strings for length {length}.")

flattened_flipflops = np.array(list(chain.from_iterable(all_valid_flipflops)))

np.savez_compressed(path + "/flipflop.npz", array=flattened_flipflops)

print(f"Saved {len(flattened_flipflops)} valid FlipFlop strings to {path}/flipflop.npz.")
