#!/usr/bin/env python
import argparse
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
#  CONFIGURATION (static) ######################################################
###############################################################################

# -- model display names -------------------------------------------------------
MODELS_INSTRUCT = {
    "qwen2.5_32B_instruct": "Qwen-2.5 32B Instruct",
    "llama3_70B_instruct" : "Llama-3 70B Instruct",
}
MODELS_COMPLETION = {
    "qwen2.5_32B" : "Qwen-2.5 32B",
    "llama3_70B"  : "Llama-3 70B",
}

# -- model families (for colouring) -------------------------------------------
MODEL_META_INSTRUCT = {
    "qwen2.5_32B_instruct": ("Qwen",   "dark"),
    "llama3_70B_instruct" : ("Llama3", "dark"),
}
MODEL_META_COMPLETION = {
    "qwen2.5_32B": ("Qwen",   "dark"),
    "llama3_70B" : ("Llama3", "dark"),
}

# colour palette: (dark, light)
COLOURS = {
    "Qwen"  : ("#006400", "#66c2a5"),
    "Llama3": ("#1f77b4", "#aec7e8"),
}

# experiment folders
ROOT_DIRS = [
    Path("results/retrieval"),
    Path("results/copying/unique"),
    Path("results/copying/nonunique"),
]

# ordered task pairs (left/backwards , right/forwards) -------------------------
TASK_PAIRS   = [
    ("UL",       "UR"),
    ("NLFirst",  "NRFirst"),
    ("NLLast",   "NRLast"),
    ("UB",       "UF"),
    ("NB",       "NF"),
]
TASK_LABELS   = [f"{a}/{b}" for a, b in TASK_PAIRS]
ORDERED_TASKS = [t for pair in TASK_PAIRS for t in pair]

# misc thresholds & regexes
MIN_LEN = 0
MAX_LEN = 30
SEED_RE = re.compile(r"seed(\d+)")
LEN_RE  = re.compile(r"_(\d+)\.jsonl$")


###############################################################################
#  ARGPARSE ####################################################################
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot retrieval+copying accuracy for instruct / completion models."
    )
    parser.add_argument(
        "--variant", "-v",
        choices=["instruct", "completion", "both"],
        default="instruct",
        help="Which group of models to plot (default: instruct)."
    )
    return parser.parse_args()


###############################################################################
#  HELPERS #####################################################################
###############################################################################
def _extract_seed(fname: str) -> int:
    m = SEED_RE.search(fname)
    return int(m.group(1)) if m else -1


def _extract_len(fname: str) -> int:
    if "len" in fname:
        return int(fname.split("_")[-3])
    elif "L" in fname:
        return int(fname.split("_")[2][1:])
    return -1


def _clean(txt: str) -> str:
    """Remove spaces and special tokens like <eos>."""
    return txt.replace(" ", "").replace("<eos>", "").strip()


###############################################################################
#  DATA GATHERING ##############################################################
###############################################################################
def _collect_records(model_key: str, prediction: bool = False) -> pd.DataFrame:
    """Walk all ROOT_DIRS and return a DataFrame with one row per prediction."""
    recs = []
    for root in ROOT_DIRS:
        if not root.exists():
            continue
        for task_dir in root.rglob("*"):
            if not task_dir.is_dir() or task_dir.name not in ORDERED_TASKS:
                continue
            model_dir = task_dir / model_key
            if not model_dir.is_dir():
                continue
            for prompt_dir in model_dir.iterdir():
                if not prompt_dir.is_dir():
                    continue
                prompt_type = prompt_dir.name
                for jf in prompt_dir.glob("*.jsonl"):
                    tgt_len = _extract_len(jf.name)
                    if tgt_len < MIN_LEN or tgt_len > MAX_LEN:
                        continue
                    seed = _extract_seed(jf.name)
                    with jf.open(encoding="utf-8") as f:
                        for ln in f:
                            j   = json.loads(ln)
                            tgt = _clean(j["target"])
                            out = _clean(
                                j.get("prediction" if prediction else "full_output", "")
                            )
                            recs.append(
                                dict(
                                    task=task_dir.name,
                                    model=model_key,
                                    prompt_type=prompt_type,
                                    seed=seed,
                                    correct=(tgt == out),
                                )
                            )
    if not recs:
        raise RuntimeError(f"No records found for {model_key}")
    return pd.DataFrame(recs)


def load_all_models(models_dict) -> pd.DataFrame:
    """Load and concatenate records for every model key passed in."""
    dfs = [
        _collect_records(m_key, prediction=("instruct" in m_key))
        for m_key in models_dict
    ]
    return pd.concat(dfs, ignore_index=True)


###############################################################################
#  METRICS #####################################################################
###############################################################################
def compute_means_stds(df: pd.DataFrame):
    """Return bar means (height) and std-devs (error) for every task × model."""
    acc_seed = (
        df.groupby(["task", "model", "prompt_type", "seed"])["correct"].mean()
    )
    acc_prompt = acc_seed.groupby(level=[0, 1, 2]).mean()
    bar_means = acc_prompt.groupby(level=[0, 1]).mean()
    bar_stds  = acc_prompt.groupby(level=[0, 1]).std(ddof=0).fillna(0)
    return bar_means, bar_stds


###############################################################################
#  PLOTTING ####################################################################
###############################################################################
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def fig_joint_accuracy(
    bar_means: pd.Series,
    bar_stds: pd.Series,
    models_dict,
    model_meta,
    outfile: str,
):
    """Five task-pairs × four bars with a second-level X-axis."""
    width, inner_gap, outer_gap = 0.18, 0.03, 0.45

    # -- compute x positions ---------------------------------------------------
    xs, grp_centres = [], []
    for g in range(len(TASK_PAIRS)):
        base = g * (4 * width + 3 * inner_gap + outer_gap)
        xs.extend([base + i * (width + inner_gap) for i in range(4)])
        grp_centres.append(base + 1.5 * width + inner_gap)

    # -- plot bars -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3))
    bar_handles, idx = {}, 0
    for (left_task, right_task) in TASK_PAIRS:
        for m_key in models_dict:
            fam, _ = model_meta[m_key]
            for variant, task in enumerate([left_task, right_task]):
                shade = 1 if variant == 0 else 0  # light shade = left/back
                colour = COLOURS[fam][shade]
                mean, std = bar_means.loc[(task, m_key)], bar_stds.loc[(task, m_key)]
                h = ax.bar(
                    xs[idx],
                    mean + std,
                    width=width,
                    capsize=3,
                    color=colour,
                    edgecolor="black",
                    linewidth=0.4,
                )
                bar_handles[(m_key, shade)] = bar_handles.get((m_key, shade), h[0])
                idx += 1

    # -- axes, ticks, legend ----------------------------------------------------
    ax.set_xticks(grp_centres, TASK_LABELS)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    legend_labels = {
        (m_key, 1): f"{models_dict[m_key]} – left/back"
        for m_key in models_dict
    } | {
        (m_key, 0): f"{models_dict[m_key]} – right/fwd"
        for m_key in models_dict
    }
    handles = [bar_handles[k] for k in legend_labels]
    labels  = [legend_labels[k] for k in legend_labels]
    ax.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize="large",
        frameon=True,
    )

    # -- second-level X-axis ----------------------------------------------------
    retrieval_centre, copying_centre = np.mean(grp_centres[:3]), np.mean(grp_centres[3:])
    y_off = 1.07
    ax.text(
        retrieval_centre,
        y_off,
        "RETRIEVAL",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=13,
        fontweight="semibold",
    )
    ax.text(
        copying_centre,
        y_off,
        "COPYING",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=13,
        fontweight="semibold",
    )

    fig.subplots_adjust(top=0.15)
    fig.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    print(f"✔ Saved plot ➜ {outfile}")


###############################################################################
#  MAIN ########################################################################
###############################################################################
def main():
    args = parse_args()

    # select models & metadata based on CLI flag --------------------------------
    if args.variant == "instruct":
        MODELS = MODELS_INSTRUCT
        MODEL_META = MODEL_META_INSTRUCT
    elif args.variant == "completion":
        MODELS = MODELS_COMPLETION
        MODEL_META = MODEL_META_COMPLETION
    else:  # both
        MODELS = MODELS_COMPLETION | MODELS_INSTRUCT
        MODEL_META = MODEL_META_COMPLETION | MODEL_META_INSTRUCT

    df_full = load_all_models(MODELS)
    means, stds = compute_means_stds(df_full)

    # optional: per-seed accuracies print-out
    table = (
        df_full.groupby(["task", "model", "seed"])["correct"]
        .mean()
        .unstack("seed")
        .loc[ORDERED_TASKS]
    )
    print("\n=== per-seed accuracies (best prompts averaged) ===")
    print(table)

    outfile = f"visualisations/combined_retrieve_copy_{args.variant}.pdf"
    fig_joint_accuracy(means, stds, MODELS, MODEL_META, outfile)


if __name__ == "__main__":
    main()
