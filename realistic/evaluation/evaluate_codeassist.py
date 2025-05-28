from __future__ import annotations
###############################################################################
#  Imports & GLOBAL CONFIG                                                    #
###############################################################################

import argparse
import json
from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

###############################################################################
#  CONSTANTS                                                                  #
###############################################################################

TASKS  = ("revert", "cherrypick")
# Base models and display names ------------------------------------------------
MODELS_BASE = {
    "llama3_8B":  "Llama‑3 8B",
    "qwen2.5_7B":  "Qwen2.5 7B",
    "qwen2.5_32B":  "Qwen2.5 32B",
    "llama3_70B": "Llama‑3 70B",
}
# Instruction‑tuned placeholders – keys match base key + "_inst"
MODELS_INST = {f"{m}_inst": f"{n}‑IT" for m, n in MODELS_BASE.items()}

DISPLAY_ORDER = list(MODELS_BASE.keys()) + list(MODELS_INST.keys())

TASK_COLOURS    = {
    "revert":     mpl.colormaps["tab10"].colors[0],
    "cherrypick": mpl.colormaps["tab10"].colors[1],
}
# Regex helpers ---------------------------------------------------------------
SEED_RE   = re.compile(r"^(?P<base>.+?)_seed(?P<seed>\d+)$")
END_TOKEN = re.compile(r"\s*<end>\s*$", re.I)

###############################################################################
#  Utility functions                                                          #
###############################################################################

def _clean_lines(text: str) -> List[str]:
    """Return stripped, non‑empty lines (trailing <end> clipped)."""
    lines = text.splitlines()
    if lines and END_TOKEN.search(lines[-1]):
        lines[-1] = END_TOKEN.sub("", lines[-1])
    return [ln.strip() for ln in lines if ln.strip()]


def _parse_instruct_response(response_text):
    ans_begin = response_text.find("<start>")
    ans_end = response_text.find("<end>")
    answer = response_text[ans_begin + len("<start>"): ans_end]
    return answer.strip()


def _parse_stem(stem: str) -> Tuple[str, int, int]:
    """Return (base_model, seed). Default seed=0 if none found."""
    stem_list = stem.split('_')
    if len(stem_list[-1]) == 4: 
        seed = int(stem_list[-1])
        length = int(stem_list[-2])
    else:
        # random seed
        seed = 99
        length = int(stem_list[-1])

    model = '_'.join(stem_list[:3]) if 'inst_' in stem else '_'.join(stem_list[:2])
    return model, seed, length


###############################################################################
#  Data aggregation                                                           #
###############################################################################

def evaluate_dir(task_dir: Path) -> list[dict]:
    """Return list of dict rows for *one* task directory."""
    rows = []
    task = task_dir.name  # revert | cherrypick
    for jsonl in task_dir.glob("*.jsonl"):
        model_key, seed, seq_len = _parse_stem(jsonl.stem)
        if seq_len > 20 or model_key != 'llama3_8B_inst':
            continue
        print(model_key, seed, seq_len)
        ok = tot = 0
        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                obj  = json.loads(line)
                full = _clean_lines(_parse_instruct_response(obj["full_answer"]))
                gold = _clean_lines(obj["gold_ans"])
                ok  += full == gold
                tot += 1
        if tot:
            rows.append({
                "task":   task,
                "model":  model_key,
                "seed":   seed,
                "correct": ok,
                "total":   tot,
                "accuracy": ok / tot,
            })
    return rows


###############################################################################
#  PLOTTING ###################################################################
###############################################################################

###############################################################################
#  PLOTTING ###################################################################
###############################################################################

def plot_accuracy(df: pd.DataFrame, *, include_inst: bool = False, save: bool = False):
    """Create grouped bar plot with mean±std across seeds.

    X‑axis = model families (base and optional *_inst*),
    coloured bars = tasks (revert / cherrypick).
    """
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    })

    # ── Compute mean ± std across seeds ─────────────────────────────────────
    stats = (
        df.groupby(["model", "task"], as_index=False)
          .agg(mean=("accuracy", "mean"), std=("accuracy", "std"))
    )

    # Ensure NaN std becomes 0 (single‑seed edge‑case)
    stats["std"].fillna(0.0, inplace=True)

    models = DISPLAY_ORDER if include_inst else list(MODELS_BASE)
    n_models = len(models)
    bar_width = 0.15

    x_indices = np.arange(n_models)
    fig, ax = plt.subplots(figsize=(8, 4))

    # For legend: keep handles only once per task
    legend_handles = {}

    # Use symmetric offsets so bars sit side‑by‑side
    task_offsets = {TASKS[0]: -bar_width/2, TASKS[1]: bar_width/2}

    for task in TASKS:
        offset = task_offsets[task]
        xpos = x_indices + offset

        heights = []
        errors  = []
        hatches = []
        faces   = []
        edges   = []
        for m in models:
            row = stats[(stats.model == m) & (stats.task == task)]
            if row.empty:
                height = 0.0
                error  = 0.0
            else:
                height = float(row["mean"].iloc[0])
                error  = float(row["std"].iloc[0])
            heights.append(height)
            errors.append(error)

            # Style decisions ------------------------------------------------
            is_base = not m.endswith("_inst")
            faces.append(TASK_COLOURS[task] if is_base else "none")
            edges.append(TASK_COLOURS[task])
            hatches.append(None if is_base else "//")

        for i, (h, e, face, edge, hatch) in enumerate(zip(heights, errors, faces, edges, hatches)):
            bar = ax.bar(
                x_indices[i] + offset,
                h,
                bar_width,
                yerr=e,
                capsize=3 if e else 0,
                facecolor=face,
                edgecolor=edge,
                hatch=hatch,
                label=task if task not in legend_handles else "_nolegend_",
            )
            legend_handles[task] = True

    
    # X‑labels -----------------------------------------------------------------
    print(legend_handles)
    x_labels = [MODELS_BASE.get(m, MODELS_INST.get(m, m)) for m in models]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=0)

    # Cosmetics ----------------------------------------------------------------
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)

    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 0.9), ncol=2)
    fig.tight_layout()

    if save:
        fig.savefig("visualisations/revert_cherrypick.pdf", bbox_inches="tight")
    plt.show()


###############################################################################
#  CLI ENTRY‑POINT ###########################################################
###############################################################################

def main(root: Path, include_inst: bool, save: bool):
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder '{root}' does not exist.")

    rows = []
    for task_dir in root.iterdir():
        if task_dir.is_dir() and task_dir.name in TASKS:
            rows.extend(evaluate_dir(task_dir))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data found – check paths, task names, or model list.")

    print(df)
    plot_accuracy(df, include_inst=include_inst, save=save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot revert vs cherrypick accuracy (bar chart)")
    parser.add_argument("--root", "-r", default="results/realistic/codeassist",
                        help="folder with 'revert/' and 'cherrypick/' sub‑dirs")
    parser.add_argument("--include-inst", action="store_true",
                        help="add empty placeholder bars for *_inst models")
    parser.add_argument("--save", action="store_true",
                        help="save PNG to script directory as 'accuracy_revert_vs_cherrypick.png'")
    args = parser.parse_args()
    main(Path(args.root), include_inst=args.include_inst, save=args.save)
