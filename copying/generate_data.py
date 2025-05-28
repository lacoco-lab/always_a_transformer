"""copy_dataset_generator.py

Generator that produces JSON‑Lines datasets for the *Copy* probing task.  
A **sample** always contains one *input* sequence together with **two** target
columns – one for the *forward* copy and one for the *backward* copy – so that
both settings of a group are stored in the *same* file:

* **unique** group → fields ``input``, ``UF``, ``UB``
* **nonunique** group → fields ``input``, ``NF``, ``NB``

The caller decides which group by passing ``group="unique"`` or
``group="nonunique"`` when instantiating :class:`CopyDatasetGenerator`.

Usage example
-------------
>>> from copy_dataset_generator import CopyDatasetGenerator
>>> gen = CopyDatasetGenerator(
...     vocab="abcde", output_dir="datasets", group="unique", seed=42)
>>> path = gen.generate_dataset(num_samples=3, length=5)
>>> print(path.read_text())
{"input": "abcde", "UF": "abcde", "UB": "edcba"}
{"input": "bedca", "UF": "bedca", "UB": "acdeb"}
{"input": "caebd", "UF": "caebd", "UB": "dbeac"}

"""
from __future__ import annotations

import random
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines

__all__ = [
    "Setting",
    "CopyDatasetGenerator",
]


class Setting(Enum):
    """Enumeration of the four copy settings."""

    UF = auto()  # Unique – Forward copying
    UB = auto()  # Unique – Backward copying
    NF = auto()  # Non‑unique – Forward copying
    NB = auto()  # Non‑unique – Backward copying

    # ------- Convenience helpers ------- #

    def is_unique(self) -> bool:  # noqa: D401 (simple is fine)
        """Return ``True`` iff setting belongs to the *unique* group."""
        return self in {Setting.UF, Setting.UB}


class CopyDatasetGenerator:
    """Generate copy‑task datasets with *grouped* targets.

    Parameters
    ----------
    vocab
        Iterable of characters/tokens used to build input strings. **Order is
        preserved** while duplicates are removed.
    output_dir
        Directory in which dataset files are created.
    group
        Either ``"unique"`` (produce *UF/UB* targets) or ``"nonunique"``
        (produce *NF/NB* targets).
    seed
        Optional deterministic RNG seed.
    """

    # ---------------- Public API ---------------- #

    def __init__(
        self,
        vocab: str,
        output_dir: str | Path,
        group: str,
        seed: Optional[int] = None,
    ) -> None:
        if group not in {"unique", "nonunique"}:
            raise ValueError("group must be either 'unique' or 'nonunique'")

        # stable‑unique vocabulary order
        self.vocab: List[str] = list(dict.fromkeys(vocab))
        if len(self.vocab) < 2:
            raise ValueError("vocab must contain at least two distinct tokens")

        self.group = group
        self.rng = random.Random(seed)
        self.seed = seed

        self.save_dir = Path(output_dir) / group
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- #
    #                         Generation                          #
    # ----------------------------------------------------------- #

    def generate_sample(self, length: int) -> Dict[str, str]:
        """Return a single dataset *record*.

        The record schema depends on the *group* chosen at construction time:

        * ``{"input": <str>, "UF": <str>, "UB": <str>}``
        * ``{"input": <str>, "NF": <str>, "NB": <str>}``
        """
        if length < 2:
            raise ValueError("length must be ≥ 2")

        if self.group == "unique":
            if length > len(self.vocab):
                raise ValueError(
                    "length cannot exceed vocabulary size for unique strings",
                )
            input_seq = self._unique_string(length)
            record = {
                "input": input_seq,
                "UF": input_seq,
                "UB": input_seq[::-1],
            }
        else:  # non‑unique group
            input_seq = self._nonunique_string(length)
            record = {
                "input": input_seq,
                "NF": input_seq,
                "NB": input_seq[::-1],
            }
        return record

    def generate_dataset(self, num_samples: int, length: int) -> Path:
        """Create *num_samples* records and write them to a JSON‑Lines file.

        Returns
        -------
        Path
            Location of the written file.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be ≥ 1")

        filename = (
            f"{self.group}_N{num_samples}_L{length}_seed{self.seed}.jsonl"
        )
        out_path = self.save_dir / filename

        with jsonlines.open(out_path, mode="w") as writer:
            for _ in range(num_samples):
                writer.write(self.generate_sample(length))

        return out_path

    # ---------------- Internal helpers ---------------- #

    def _unique_string(self, length: int) -> str:
        """Return a *unique* token string of the requested length."""
        # ``random.sample`` returns *k* unique elements from the population.
        return "".join(self.rng.sample(self.vocab, length))

    def _nonunique_string(self, length: int) -> str:
        """Return a *challenging* non‑unique string for the copy task.

        Strategy
        ^^^^^^^^
        * **length ≤ 5** → behave exactly like the original reference
          implementation (at least one duplication – nothing fancy).
        * **length > 5** → construct a high‑duplication sequence:

          #. Choose a *token pool* ``P`` whose size grows with ``length``:
             ``|P| = min(len(vocab), max(2, length // 3))`` but never larger
             than half of the requested length (so that each can repeat twice).
          #. Ensure every element of ``P`` appears **at least twice** in the
             sequence.
          #. Fill any remaining positions with additional random draws from
             ``P`` (duplicates are therefore guaranteed).
          #. Finally, apply a Fisher‑Yates shuffle to destroy any ordering
             clues the model might exploit.
        """
        # --- short & simple ------------------------------------------------ #
        if length <= 5:
            # identical to the previous simple baseline
            while True:
                tokens = [self.rng.choice(self.vocab) for _ in range(length)]
                if len(set(tokens)) < length:
                    return "".join(tokens)
                # ensure duplication by force
                dup_idx = self.rng.randint(0, length - 2)
                tokens[-1] = tokens[dup_idx]
                return "".join(tokens)

        # --- tougher generation ------------------------------------------- #
        # 1. determine pool size (each token will appear at least twice)
        max_pool = length // 2  # every token must fit twice
        pool_size = min(max_pool, max(2, length // 3,))
        pool_size = min(pool_size, len(self.vocab))

        pool = self.rng.sample(self.vocab, pool_size)

        # 2. place each pool token twice
        tokens: List[str] = []
        for tok in pool:
            tokens.extend([tok, tok])

        # 3. fill the rest with random tokens *from the same pool*
        remaining = length - len(tokens)
        tokens.extend(self.rng.choices(pool, k=remaining))

        # 4. in‑place shuffle (Fisher‑Yates via random.shuffle)
        self.rng.shuffle(tokens)

        return "".join(tokens)

# -------------------- Demo -------------------- #
if __name__ == "__main__":
    import string

    VOCAB = string.ascii_letters
    for seed_num in [121, 122, 123]:
        gen_unique = CopyDatasetGenerator(
            vocab=VOCAB, output_dir="datasets/copying", group="unique", seed=seed_num)
        gen_nonunique = CopyDatasetGenerator(
            vocab=VOCAB, output_dir="datasets/copying", group="nonunique", seed=seed_num)

        for length in [100]:
            # print("Writing UNIQUE dataset…", gen_unique.generate_dataset(1500, length))
            print("Writing NONUNIQUE dataset…", gen_nonunique.generate_dataset(1500, length))
