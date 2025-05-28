#!/usr/bin/env python3
"""generate_dataset.py
--------------------------------
Create synthetic (instruction, digit) sequences ending with a *query* instruction
separated from the rest of the string by the delimiter `||`.

Example line:
    {"config": "first_right", "input": "i0w7r3w5q9||w", "golden_answer": "3"}

Key guarantees
^^^^^^^^^^^^^^
* **Four retrieval configs** – first_right, first_left, last_right, last_left
* **≈10% query‑only variants** – reuse the same body but change only the final query instruction
* **Unique inputs** – all `input` strings are distinct
* **JSONL output** – one JSON object per line in the outfile
* **Progress bar** – shows generation progress via `tqdm` if available
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ALPHABET: str = string.ascii_lowercase
DIGITS: str = string.digits
DELIMITER: str = "||"
CONFIGS = ["first_right", "first_left", "last_right", "last_left"]


def _choose_query_positions(n_pairs: int) -> List[int]:
    valid = list(range(1, n_pairs - 1))
    k = random.randint(1, min(3, len(valid)))
    return sorted(random.sample(valid, k=k))


def _build_sequence(n_pairs: int, query_instr: str, query_positions: List[int]) -> str:
    parts: List[str] = []
    for idx in range(n_pairs):
        instr = query_instr if idx in query_positions else random.choice(ALPHABET)
        digit = random.choice(DIGITS)
        parts.append(instr + digit)
    return "".join(parts)


def _answer_index(query_positions: List[int], config: str) -> int:
    """Select the correct index based on config:
    - first_left:  neighbor left of the first query occurrence
    - first_right: digit inside the first query pair
    - last_left:   neighbor left of the last query occurrence
    - last_right:  digit inside the last query pair
    """
    first, last = query_positions[0], query_positions[-1]
    if config == "first_left":
        return first - 1
    elif config == "first_right":
        return first
    elif config == "last_left":
        return last - 1
    elif config == "last_right":
        return last
    else:
        raise ValueError(f"Unknown config: {config}")


def _generate_sample(min_pairs: int, max_pairs: int, config: str) -> Tuple[Dict, List[str]]:
    if min_pairs < 3:
        min_pairs = 3
    n_pairs = random.randint(min_pairs, max_pairs)

    query_instr = random.choice(ALPHABET)
    positions = _choose_query_positions(n_pairs)

    seq_str = _build_sequence(n_pairs, query_instr, positions)
    pairs = [seq_str[i:i + 2] for i in range(0, len(seq_str), 2)]

    # Recompute actual occurrences of the query instruction in the generated sequence
    actual_positions = [idx for idx, pair in enumerate(pairs) if pair[0] == query_instr]

    ans_idx = _answer_index(actual_positions, config)
    golden = pairs[ans_idx][1]

    sample = {
        "config": config,
        "input": f"{seq_str}{DELIMITER}{query_instr}",
        "golden_answer": golden,
    }
    return sample, pairs


def _make_query_variant(base_sample: Dict, pairs: List[str], seen: Set[str]) -> Dict | None:
    body, old_query = base_sample["input"].split(DELIMITER)
    config = base_sample["config"]

    instr_positions: Dict[str, List[int]] = {}
    for idx, pair in enumerate(pairs):
        instr = pair[0]
        # skip same as old query and edge neighbors invalid for left variants
        if instr == old_query:
            continue
        instr_positions.setdefault(instr, []).append(idx)

    if not instr_positions:
        return None

    new_query, q_positions = random.choice(list(instr_positions.items()))
    ans_idx = _answer_index(q_positions, config)
    golden = pairs[ans_idx][1]

    new_input = f"{body}{DELIMITER}{new_query}"
    if new_input in seen:
        return None

    return {"config": config, "input": new_input, "golden_answer": golden}


def _save_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(samples)} samples to {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strings following generic rules")
    parser.add_argument("-min", "--min-length", type=int, default=4,
                        help="Minimum number of <instruction,digit> pairs (default: 4)")
    parser.add_argument("-max", "--max-length", type=int, required=True,
                        help="Maximum number of <instruction,digit> pairs (required)")
    parser.add_argument("-n", "--samples", type=int, default=1,
                        help="Total number of samples to generate (default: 1)")
    parser.add_argument("-c", "--config", choices=CONFIGS,
                        help="Retrieval configuration (overrides --random-config)")
    parser.add_argument("--random-config", action="store_true",
                        help="Pick a random retrieval configuration for each base sample")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help="Output file path (JSONL). Prints to console regardless)")
    parser.add_argument("-s", "--seed", type=int, default=51,
                        help="RNG seed for reproducibility (default: 51)")
    args = parser.parse_args()

    if not args.config and not args.random_config:
        args.config = "first_right"

    random.seed(args.seed)

    dup_count = max(1, round(args.samples * 0.10)) if args.samples >= 10 else 0
    base_count = args.samples - dup_count

    bases: List[Tuple[Dict, List[str]]] = []
    seen: Set[str] = set()

    total = args.samples
    pbar = tqdm(total=total, desc="Generating samples", unit="sample", file=sys.stderr) if tqdm else None

    attempts = 0
    while len(bases) < base_count:
        attempts += 1
        if attempts > base_count * 20:
            raise RuntimeError("Unable to generate enough unique base samples; consider increasing length limits.")
        cfg = random.choice(CONFIGS) if args.random_config else args.config
        sample, pairs = _generate_sample(args.min_length, args.max_length, cfg)
        if sample["input"] in seen:
            continue
        bases.append((sample, pairs))
        seen.add(sample["input"])
        if pbar:
            pbar.update()

    dataset: List[Dict] = [b for b, _ in bases]

    attempts = 0
    while len(dataset) < args.samples:
        attempts += 1
        if attempts > dup_count * 20:
            raise RuntimeError("Unable to create enough query-only variants; try different parameters.")
        base_sample, pairs = random.choice(bases)
        variant = _make_query_variant(base_sample, pairs, seen)
        if variant is None:
            continue
        dataset.append(variant)
        seen.add(variant["input"])
        if pbar:
            pbar.update()

    if pbar:
        pbar.close()

    _save_jsonl(dataset, Path(args.outfile))


if __name__ == "__main__":
    main()
