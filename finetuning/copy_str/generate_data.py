#!/usr/bin/env python3
import argparse
import json
import random
import string
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate JSONL samples of unique-character strings with a progress bar."
    )
    p.add_argument('-min', '--min-length', type=int, required=True,
                   help="Minimum length of each string (inclusive)")
    p.add_argument('-max', '--max-length', type=int, required=True,
                   help="Maximum length of each string (inclusive)")
    p.add_argument('-n', '--samples', type=int, required=True,
                   help="Number of unique samples to generate")
    p.add_argument('-o', '--outfile', type=Path, required=True,
                   help="Path to output JSONL file")
    p.add_argument('-s', '--seed', type=int, default=51,
                   help="Random seed (optional)")
    p.add_argument('-d', '--delimiter', type=str, default='>',)
    p.add_argument('-c', '--config', type=str, default='normal',)
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare character pool
    all_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    max_unique = len(all_chars)

    # Validate lengths
    if args.max_length > max_unique:
        sys.exit(f"Error: max-length ({args.max_length}) exceeds number of unique characters ({max_unique}).")
    if args.min_length < 1 or args.min_length > args.max_length:
        sys.exit("Error: min-length must be >=1 and <= max-length.")

    # Seed RNG
    random.seed(args.seed)

    # Generate unique samples with progress bar
    samples_set = set()
    pbar = tqdm(total=args.samples, desc="Generating samples", unit="sample")
    while len(samples_set) < args.samples:
        length = random.randint(args.min_length, args.max_length)
        s = ''.join(random.sample(all_chars, length)) + args.delimiter
        if s not in samples_set:
            samples_set.add(s)
            pbar.update(1)
    pbar.close()

    # Convert to list and shuffle order
    samples = list(samples_set)
    random.shuffle(samples)

    # Ensure output directory exists
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSONL
    with args.outfile.open('w', encoding='utf-8') as f:
        for s in samples:
            json.dump({
                'input': s, 
                'golden_answer': s.replace(args.delimiter, '') if args.config == 'normal' else s.replace(args.delimiter, '')[::-1]}, 
                f)
            f.write('\n')

    print(f"Successfully generated {args.samples} unique samples to {args.outfile}")


if __name__ == '__main__':
    main()
