import sys
import numpy as np
from flipflop_generator import generate_all_valid_flipflops, validate_flip_flop, generate_flip_flop, generate_flip_flop_with_distance


"""
This is a script to generate valid FlipFlop strings in the needed range.
Provide ranges as command line arguments by running:

python flipflop/generate_flipflop.py 4 20

where 4 is the lower boundary and 20 is the upper boundary.

Strings are saved as a numpy file in the datasets/flipflop.
"""


length, w_idx, limit = sys.argv[1:]
path = "datasets/flipflop"

def generate_from_to(start_length, finish_length):
    """
    Generate all valid flipflops in the range of strings from start_length to finish_length.
    :param start_length: int
    :param finish_length: int
    :return: list of flipflops
    """

    all_valid_flipflops = []

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

    return all_valid_flipflops


def generate_with_density(length, pw, pr, limit=1000):
    """
    Generate all valid flipflops in the range of strings of a certain length with probability density.
    :param length:  int
    :param pw: float
    :param pr: float
    :return: list of flipflops
    """

    # Beware about the limit parameter; 1000 can only be generated starting from length 14
    # 100 from the length 10

    all_valid_flipflops = []

    valid_flipflops_count = 0
    while valid_flipflops_count < limit:
        flipflop = generate_flip_flop(length, pw, pr)

        try:
            validate_flip_flop(flipflop)
            if flipflop is all_valid_flipflops:
                continue
            all_valid_flipflops.append(flipflop)
            valid_flipflops_count += 1
        except AssertionError:
            continue

    print(f"Generated {valid_flipflops_count} strings for length {length} with pw of {pw} and pr of {pr}.")

    return all_valid_flipflops


def generate_with_distance_w(length, w_idx, limit=1000):
    """
    Generate all valid flipflops with a certain distance to w
    :param length: int
    :param w_idx: int, index of the last desired write
    :return: list of flipflops
    """

    # Beware about the limit parameter
    # Might send us into an infinite loop

    all_valid_flipflops = []

    valid_flipflops_count = 0
    while valid_flipflops_count < limit:
        flipflop = generate_flip_flop_with_distance(length, w_idx)

        try:
            validate_flip_flop(flipflop)
            if flipflop is all_valid_flipflops:
                continue
            all_valid_flipflops.append(flipflop)
            valid_flipflops_count += 1
        except AssertionError:
            continue

    print(f"Generated {valid_flipflops_count} strings for length {length} with idx of w {w_idx}.")

    return all_valid_flipflops


all_valid_flipflops = generate_with_distance_w(int(length), int(w_idx), int(limit))
save_path = path + f"/distance/s2/flipflop_{length}_w{w_idx}.txt"
np.savetxt(save_path, all_valid_flipflops, delimiter='\n', fmt='%s')

print(f"Saved {len(all_valid_flipflops)} valid FlipFlop strings to {save_path}/flipflop_{length}_w{w_idx}.txt.")
