from __future__ import annotations

import string
import random
import jsonlines
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Optional



__all__ = [
    "Setting",
    "RetrievalDatasetGenerator",
]


class Setting(Enum):
    """Enumeration of the six induction settings."""

    UL = auto()  # Unique – Left
    UR = auto()  # Unique – Right
    NLFirst = auto()  # Non‑unique, first occurrence, Left neighbour
    NRFirst = auto()  # Non‑unique, first occurrence, Right neighbour
    NLLast = auto()  # Non‑unique, last  occurrence, Left neighbour
    NRLast = auto()  # Non‑unique, last  occurrence, Right neighbour

    # ---------------- Convenience helpers ---------------- #

    def is_unique(self) -> bool:
        return self in {Setting.UL, Setting.UR}

    def neighbour_offset(self) -> int:
        return -1 if "L" in self.name else 1

    def focus_index(self) -> int:
        # Which q occurrence is relevant: first (0), last (-1), or only (0)
        if self in {Setting.NLFirst, Setting.NRFirst}:
            return 0
        if self in {Setting.NLLast, Setting.NRLast}:
            return -1
        return 0  # UL / UR


# ---------------- Dataset generator ---------------- #


class RetrievalDatasetGenerator:
    """Generator for synthetic induction datasets.

    Parameters
    ----------
    vocab
        Iterable of characters/tokens used to build strings.
    output_dir
        Root directory where dataset files are written.
    group
        Either ``"unique"`` (generates UL/UR) or ``"nonunique"``
        (generates NL*/NR*).
    seed
        Optional RNG seed for reproducibility.
    """

    UNIQUE_GROUP = {Setting.UL, Setting.UR}
    NONUNIQUE_GROUP = {
        Setting.NLFirst,
        Setting.NRFirst,
        Setting.NLLast,
        Setting.NRLast,
    }

    def __init__(
        self,
        vocab: str,
        output_dir: str | Path,
        group: str,
        seed: Optional[int] = None,
    ) -> None:
        if group not in {"unique", "nonunique"}:
            raise ValueError("group must be either 'unique' or 'nonunique'")

        self.vocab: List[str] = list(dict.fromkeys(vocab))  # stable unique
        if len(self.vocab) < 3:
            raise ValueError("vocab must contain at least three distinct tokens")

        self.group = group  # "unique" or "nonunique"
        self.rng = random.Random(seed)
        self.seed = seed

        self.save_dir = Path(output_dir) / group
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Public API ---------------- #

    def generate_sample(self, length: int) -> Dict[str, str]:
        """Return a dict with the context and all relevant target fields."""
        if length < 4:
            raise ValueError("length must be ≥ 4")

        # Randomly choose the *query* token (distinct from other tokens)
        q = self.rng.choice(self.vocab)
        non_q_vocab = [tok for tok in self.vocab if tok != q]

        context_len = length - 1  # exclude trailing q in position n‑1

        # Determine positions of q inside the context (indices 0..context_len‑1)
        if self.group == "unique":
            q_positions = [self._sample_q_position_unique(context_len)]
        else:
            q_positions = self._sample_q_positions_non_unique(context_len)

        # Build the token sequence
        tokens = [self.rng.choice(non_q_vocab) for _ in range(context_len)]
        for pos in q_positions:
            tokens[pos] = q
        tokens.append(q)  # x_n at the end

        # Build flat target fields
        record: Dict[str, str] = {
            "input_no_space": "".join(tokens),
            "input_with_space": "".join(tokens[:-1]) + "|" + tokens[-1],
        }

        if self.group == "unique":
            record["UL"] = tokens[q_positions[0] - 1]
            record["UR"] = tokens[q_positions[0] + 1]
        else:
            first_idx, last_idx = q_positions[0], q_positions[-1]
            record["NLFirst"] = tokens[first_idx - 1]
            record["NRFirst"] = tokens[first_idx + 1]
            record["NLLast"] = tokens[last_idx - 1]
            record["NRLast"] = tokens[last_idx + 1]

        return record

    def generate_dataset(
        self,
        num_samples: int,
        length: int
    ) -> Path:
        """Generate *num_samples* examples and write them as JSON‑Lines."""
        if num_samples < 1:
            raise ValueError("num_samples must be ≥ 1")

        out_path = self.save_dir / f"sample_{num_samples}_len_{length}_seed_{self.seed}.jsonl"

        with jsonlines.open(out_path, mode="w") as writer:
            for _ in range(num_samples):
                writer.write(self.generate_sample(length))

        return out_path

    # -------------- Sampling helpers -------------- #

    def _sample_q_position_unique(self, context_len: int) -> int:
        """Return a single index such that both left and right neighbours exist."""
        # Any index in 1 .. context_len‑2 inclusive is safe
        return self.rng.randint(1, context_len - 2)

    def _sample_q_positions_non_unique(self, context_len: int) -> List[int]:
        """Return ≥2 sorted unique positions for q, with none consecutive."""
        # choose how many positions to sample
        max_positions = max(2, context_len // 5)
        k = self.rng.randint(2, max_positions)

        candidates = list(range(2, context_len-1))

        # keep sampling until no two picks are neighbours
        while True:
            picks = self.rng.sample(candidates, k)
            picks.sort()
            # check for any consecutive indices
            has_adjacent = any(b - a == 1 for a, b in zip(picks, picks[1:]))
            if not has_adjacent:
                return picks


# -------------------- Demo -------------------- #
if __name__ == "__main__":

    # Change to get different datasets
    seed_list = [20, 21, 22]
    groups = ['unique', 'nonunique']
    lengths = [20, 30, 40]
    VOCAB = string.ascii_letters + string.digits

    # Generating data to test on
    for seed in seed_list:
        for length in lengths:
            for grp in groups:
                gen = RetrievalDatasetGenerator(
                    vocab=VOCAB,
                    output_dir="datasets/retrieval",
                    group=grp,
                    seed=seed,
                )
                # For smaller few shot datasets, reduce num_samples
                path = gen.generate_dataset(num_samples=1500, length=length)
                print(f"Wrote {grp} dataset → {path}")
