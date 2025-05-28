#!/usr/bin/env python
# plot_neurips_tasks_single.py
# One‑figure summary for all retrieval & copy tasks.
# Author: <you>
# ----------------------------------------------------------
"""Generate a single bar plot that summarises accuracy on all eight tasks:

(UL, UR)  ;;  (UB, UF)  ;;  (NLFirst, NRFirst)  ;;  (NLLast, NRLast)

– Unique tasks (UL, UR, UB, UF) are known to attain perfect accuracy, so their
  bars are fixed to 1 · 0.
– Non‑unique tasks (NL/NR‑First/Last) use *adversarial* split accuracy only.
– The distance between bars that belong to the same family is smaller than the
  distance between families, giving clear visual grouping.

The script expects adversarial‑split results for the NL*/NR* tasks under the
folder  ``results/finetuning/flipflop/*.jsonl`` (following the same layout as
in the earlier script). If any file is missing, the corresponding bar is shown
hatched in light grey.
"""
# ----------------------------------------------------------
import json, math, pathlib, statistics
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.figsize": (9, 3),
        "font.size": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
    }
)

# ----------------------------------------------------------
# 1.  Task groups & layout
# ----------------------------------------------------------
GROUPS = [
    ("UB", "UF"),          # Copy (unique)
    ("NB", "NF"),          # Copy (unique)
    ("UL", "UR"),          # Induction (unique)
    ("NLFirst", "NRFirst"),  # Non‑unique first
    ("NLLast",  "NRLast"),   # Non‑unique last
]

# x‑coordinates: small gap (0.35) inside each pair, large gap (0.9) between pairs
_x = []
cur = 0.0
for g in GROUPS:
    _x.extend([cur, cur + 0.25])
    cur += 0.25 + 0.4  # leave big gap before next pair
X_POS = np.array(_x)
BAR_W = 0.22

# Colours (first two in the default tableau cycle)
IN_CRASP, OUT_CRASP = 'green', 'red'

# Mapping from substrings in file names → task label
PATTERNS = {
    r"first_left":  "NLFirst",
    r"first_right": "NRFirst",
    r"last_left":   "NLLast",
    r"last_right":  "NRLast",
    r"nb":  "NB",
    r"nf":  "NF",
}

# ----------------------------------------------------------
# 2.  Helpers ----------------------------------------------------------

def task_from_filename(name: str):
    for pat, task in PATTERNS.items():
        if pat in name:
            return task
    return None


def read_accuracy(file_list: list) -> float:
    total, correct = 0, 0
    for jsonl_path in file_list:
        with jsonl_path.open() as f:
            for line in f:
                entry = json.loads(line)
                total += 1
                if 'correct' in entry:
                    correct += bool(entry.get("correct"))
                else: 
                    prediction = entry.get('prediction', 'Nothing')
                    target = entry.get('target', 'Something')    
                    correct += (prediction == target)
    return correct / total if total else float("nan")


def load_adv_accuracies(data_dir: pathlib.Path):
    """Return dict task → adversarial accuracy (nan if missing)."""
    from collections import defaultdict
    acc = {t: float("nan") for t in PATTERNS.values()}
    task_fps = defaultdict(list)
    for fp in data_dir.glob("**/*.jsonl"):
        print(fp.name)
        if 'ood' in fp.name:
            continue
        task = task_from_filename(fp.stem)
        if task is None:
            continue
        print(task, fp)
        task_fps[task].append(fp)
    for task, file_list in task_fps.items():
        acc[task] = read_accuracy(file_list)
    return acc

# ----------------------------------------------------------
# 3.  Main ----------------------------------------------------------

def main(res_dir="results/finetuning/", save="visualisations/finetune.pdf"):
    res_path = pathlib.Path(res_dir)
    adv_acc = load_adv_accuracies(res_path)

    labels = sum(GROUPS, ())  # flatten
    values = []
    colours = []
    for lbl in labels:
        if 'Last' not in lbl and lbl not in ['NB', 'NF']:
            values.append(adv_acc.get(lbl, 1.0))
            colours.append(IN_CRASP)
        else:
            values.append(adv_acc.get(lbl, 0))
            colours.append(OUT_CRASP)
    # ------------------------------------------------------
    fig, ax = plt.subplots()

    for x, v, col in zip(X_POS, values, colours):
        if math.isnan(v):
            ax.bar(x, 1.0, BAR_W, color="lightgrey", hatch="///", alpha=0.5)
        else:
            ax.bar(x, v, BAR_W, color=col)

    # Axis cosmetics
    ax.set_xticks(X_POS)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.02)

    # Legend proxies
    ax.bar(0, 0, BAR_W, color=IN_CRASP, label="Task In C-RASP[Pos]")
    ax.bar(0, 0, BAR_W, color=OUT_CRASP, label="Task not In C-RASP[Pos]")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=13)

    fig.tight_layout()
    fig.savefig(save)
    print(f"Saved → {save}")


if __name__ == "__main__":
    main()
