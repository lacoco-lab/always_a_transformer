from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Random helpers
# ────────────────────────────────────────────────────────────────────────────────

def random_hash(length: int = 7) -> str:
    return "".join(random.choices("0123456789abcdef", k=length))


actions = [
    "Add", "Remove", "Fix", "Refactor", "Update", "Improve",
    "Optimize", "Document", "Cleanup", "Merge", "Revert", "Bug fix"
]

topics = [
    "authentication", "API", "search", "database", "UI", "CLI",
    "logging", "tests", "deployment", "config", "caching"
]

def random_commit_msg() -> str:
    return f"{random.choice(actions)} {random.choice(topics)}"

# ────────────────────────────────────────────────────────────────────────────────
# Git task generation
# ────────────────────────────────────────────────────────────────────────────────

def generate_git_example(n_commits: int = 5) -> dict:
    """
    Build one JSON‑serialisable example for *both* tasks.

    ─ Commit list is HEAD‑first (newest → oldest), just like `git log`.
    ─ We mark an *anchor* somwhere in the middle with '>>>'.
    ─   •  Revert task  : return the commits **above** the anchor, unchanged order.
        (newest → oldest, the order you’d hand to `git revert`)
    ─   •  Cherry‑pick : return those same commits but **oldest → newest**,
        the order you’d feed to `git cherry‑pick`.
    """
    if n_commits < 3:
        raise ValueError("need ≥ 3 commits so both halves are non‑empty")

    commits = [f"{random_hash()} {random_commit_msg()}" for _ in range(n_commits)]

    # pick an anchor that leaves at least one commit on each side
    snippet = "\n".join(commits)

    # the commits *above* the anchor are the ones we’ll revert / cherry‑pick
    target_lines = commits         # newest → oldest

    answer_revert      = "\n".join(target_lines)              # same order
    answer_cherry_pick = "\n".join(reversed(target_lines))    # oldest → newest

    return {
        "snippet":      snippet,
        "revert":       answer_revert,
        "cherrypick":  answer_cherry_pick,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Main CLI
# ────────────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 1000, out_dir: Path | str = "./", depth: int = 10, seed: int = 2025) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_dir = out_dir / "codeassist"

    for length in [depth, depth+5, depth+10]:
        git_path = git_dir / f"git_tasks_{length}_{seed}.jsonl"
        with git_path.open("w", encoding="utf-8") as git_file:
            for _ in range(n_samples):
                example = generate_git_example(length)
                git_file.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"✓ Wrote {git_path} with {n_samples} examples each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for directional retrieval tasks.")
    parser.add_argument("--n_samples", type=int, default=1500, help="Number of *histories* per dataset file")
    parser.add_argument("--out_dir", type=str, default="datasets/realistic/", help="Output directory")
    parser.add_argument("--commits", type=int, default=20, help="number of commits in history")
    args = parser.parse_args()

    for seed in [2025, 2026, 2027]:
        main(n_samples=args.n_samples, out_dir=args.out_dir, depth=args.commits, seed=seed)
