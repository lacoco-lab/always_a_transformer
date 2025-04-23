#!/usr/bin/env python3
"""
make_ner_name_sanitised_dataset.py

Create a 1 500-sample dataset from IOB-tagged TSV files, replacing every
PERSON mention with unique first-only or last-only aliases.

Each sample = {
    "names"     : ["Eslak Jorran", "Vezwi Pramon", …],   # unique across dataset
    "text_first": "<orig text with PERSON mentions → first names>",
    "text_last" : "<orig text with PERSON mentions → last names>",
    "text_orig" : "<original text, tokens concatenated>"
}
"""

import jsonlines
import random
import re
import string
from pathlib import Path
from typing import Iterator, List, Tuple

# --------------------------------------------------------------------------- #
#                 1.  ultra-rare deterministic name generator                 #
# --------------------------------------------------------------------------- #

random.seed(42)  # full determinism for reproducible datasets

VOWELS = "aeiouy"
CONSONANTS = "".join(set(string.ascii_lowercase) - set(VOWELS))


def _make_syllable() -> str:
    """Return a CVC (consonant-vowel-consonant) syllable like 'vez' or 'qum'."""
    return random.choice(CONSONANTS) + random.choice(VOWELS) + random.choice(CONSONANTS)


def _make_word(n_syllables: int = 2) -> str:
    """Compose a capitalised pseudo-word with *n_syllables* CVC blocks."""
    return "".join(_make_syllable() for _ in range(n_syllables)).capitalize()


class NameFactory:
    """
    Yield an endless supply of unique (first, last) pairs where
    * every first name is unique across all pairs,
    * every last  name is unique across all pairs.
    """

    def __init__(self) -> None:
        self.used_first: set[str] = set()
        self.used_last: set[str] = set()

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return self

    def __next__(self) -> Tuple[str, str]:
        first = last = ""
        # guarantee uniqueness of both parts
        while not first or first in self.used_first:
            first = _make_word()
        while not last or last in self.used_last:
            last = _make_word()
        self.used_first.add(first)
        self.used_last.add(last)
        return first, last


NAME_STREAM = iter(NameFactory())

# --------------------------------------------------------------------------- #
#           2.  utilities for parsing TSV + assembling replacement text       #
# --------------------------------------------------------------------------- #


PUNCT_NEEDS_NO_SPACE_BEFORE = set(".,;:!?)’”")
PUNCT_NEEDS_NO_SPACE_AFTER = set("“‘(")


def join_tokens(tokens: List[str]) -> str:
    """
    Re-assemble raw tokens into a readable string
    (simple English rules: no space before common punctuation,
    no space after opening quotes/brackets).
    """
    words: List[str] = []
    
    for tok in tokens:
        if words and tok not in PUNCT_NEEDS_NO_SPACE_BEFORE:
            # normal word or leading punctuation after a word → prepend space
            if words[-1] not in PUNCT_NEEDS_NO_SPACE_AFTER:
                words.append(" ")
        if tok in ["Mr", "Ms", "Dr", "Jr", "Sr", "St", "\"", "\'", "“", "‘", "/"]:
            continue
        words.append(tok)

    # If the entry is too long, truncate to 1000 words.
    if len(words) > 1000:
        paragraph = "".join(words[:1150])
        sentence_list = paragraph.split('.')
        return '.'.join(sentence_list[:-1])
    return ''.join(words)


def iter_person_spans(labels: List[str]) -> Iterator[Tuple[int, int]]:
    """
    Yield (start_idx, end_idx) token indices for contiguous PERSON spans in a
    label sequence (IOB2).  *end_idx* is exclusive.
    """
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == "B-Person" or labels[i] == "B-PER" or labels[i] == "B-PERSON":
            start = i
            i += 1
            while i < n and labels[i].startswith("I-"):
                i += 1
            yield start, i
        else:
            i += 1


def replace_persons(
    tokens: List[str],
    labels: List[str],
    use_first: bool = True,
    assigned_names: List[Tuple[str, str]] | None = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Return a new text where PERSON spans are replaced by either first-only
    or last-only alias.  If *assigned_names* is given, re-use those aliases
    (so that text_first & text_last line up).  Otherwise pull fresh names.
    """
    out_tokens: List[str] = []
    names_used: List[Tuple[str, str]] = assigned_names or []
    name_iter = iter(names_used) if assigned_names else NAME_STREAM

    i = 0
    for span_start, span_end in iter_person_spans(labels):
        # copy tokens before current span
        out_tokens.extend(tokens[i:span_start])

        # get or generate alias
        if assigned_names:
            first, last = next(name_iter)
        else:
            first, last = next(NAME_STREAM)
            names_used.append((first, last))

        out_tokens.append(first if use_first else last)
        i = span_end

    # copy remaining tail
    out_tokens.extend(tokens[i:])
    return join_tokens(out_tokens), names_used


# --------------------------------------------------------------------------- #
#                      3.  one-file-→-one-sample converter                    #
# --------------------------------------------------------------------------- #


def load_tokens_labels(tsv_path: Path) -> Tuple[List[str], List[str]]:
    """Read a TSV file ⇒ ([tokens], [labels]).  Assumes <token><TAB or space><label>."""
    tokens, labels = [], []
    with tsv_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:  # blank line in between sentences
                continue
            # robust split: last field = label
            *token_parts, label = re.split(r"\s+", ln)
            token = " ".join(token_parts)
            tokens.append(token)
            labels.append(label)
    return tokens, labels


def file_to_sample(tsv_path: Path) -> dict:
    """Turn one TSV file into one dataset dict as specified."""
    tokens, labels = load_tokens_labels(tsv_path)
    text_orig = join_tokens(tokens)

    # First pass: generate aliases + first-name replacements
    text_first, names_pairs = replace_persons(tokens, labels, use_first=True)

    # Second pass: reuse same aliases but last-name replacements
    text_last, _ = replace_persons(tokens, labels, use_first=False, assigned_names=names_pairs)

    # pack names as "First Last" strings
    names_list = [f"{f} {l}" for f, l in names_pairs]

    return {
        "names": names_list,
        "text_first": text_first,
        "text_last": text_last,
        "text_orig": text_orig,
    }


# --------------------------------------------------------------------------- #
#                         4.  dataset builder / CLI                           #
# --------------------------------------------------------------------------- #


def build_dataset(tsv_dir: Path, max_samples: int = 1_500) -> List[dict]:
    """
    Walk *tsv_dir*, convert up to *max_samples* files, return list[dict].
    Files are processed in lexical order for determinism.
    """
    samples: List[dict] = []
    for tsv_path in sorted(tsv_dir.glob("*.tsv")):
        if len(samples) >= max_samples:
            break
        samples.append(file_to_sample(tsv_path))
    return samples


if __name__ == "__main__":
    # Download and give the path to the en-worldwide-newswire dataset
    # The folder should have the files listed here - https://github.com/stanfordnlp/en-worldwide-newswire/tree/main/processed_annotated

    tsv_dir = Path('en-worldwide-newswire/processed_annotated')
    out_file = Path('datasets/people_names/dataset.jsonl')
    samples = build_dataset(tsv_dir, 1_500)
    # Write the samples to a JSONL file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_file, mode='w') as writer:
        writer.write_all(samples)

    # Can create more complicated dataset with the same entry and different names if required. 
    print(f"✔  Wrote {len(samples):,} samples to {out_file}")
