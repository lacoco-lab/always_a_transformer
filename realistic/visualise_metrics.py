#!/usr/bin/env python
"""
Model-variant comparison with worst-seed bars and error bars from other seeds.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# ---------------------------------------------------------------------------
# CONFIGURE FILE PATH --------------------------------------------------------
# ---------------------------------------------------------------------------
JSON_PATH = Path("results/realistic/loremipsum/analysis/model_comparison.json") 

# ---------------------------------------------------------------------------
# GLOBAL STYLE --------------------------------------------------------------
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",  # NeurIPS default
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    }
)
# ---------------------------------------------------------------------------
# UTILS ---------------------------------------------------------------------
# ---------------------------------------------------------------------------

def lighten_color(color: str, amount: float = 0.5) -> str:
    """Return a lighter shade of *color* by mixing with white."""
    c = np.array(to_rgb(color))
    white = np.ones_like(c)
    return tuple(c + (white - c) * amount)

# ---------------------------------------------------------------------------
# HELPER --------------------------------------------------------------------
# ---------------------------------------------------------------------------
def mean_and_range(vals: List[float]) -> Tuple[float, float, float]:
    """
    Returns mean, err_down, err_up for asymmetric error bars.
    """
    if not vals:                       # no seeds? return zeros
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    err_down = mean - min(vals)
    err_up   = max(vals) - mean
    return mean, err_down, err_up


def gather_metrics(exps: Dict[str, Dict]) -> Tuple[float, float, float, float]:
    """
    From all seeds of one variant produce:
        det_mean, nd_mean, det_err_down, det_err_up, nd_err_down, nd_err_up
    """
    det_vals = []
    nd_vals  = []
    for m in exps.values():
        det_vals.append(m["avg_deterministic_similarity"])
        nd_vals.append(m["avg_non_deterministic_similarity"])

    det_mean, det_dn, det_up = mean_and_range(det_vals)
    nd_mean,  nd_dn,  nd_up  = mean_and_range(nd_vals)
    return det_mean, nd_mean, det_dn, det_up, nd_dn, nd_up

# ---------------------------------------------------------------------------
# MODEL FAMILY CONFIG -------------------------------------------------------
# ---------------------------------------------------------------------------
FAMILIES: Dict[str, Dict] = {
    "LLama3": {
        "variants": [
            "llama3_8B",
            "llama3_8B_Instruct",
            "llama3_70B",
            "llama3_70B_Instruct",
        ],
        "color": "#1f77b4",
        "title": "LLama 3",
    },
    "Qwen2.5": {
        "variants": [
            "qwen2.5_7B",
            "qwen2.5_7B_Instruct",
            "qwen2.5_32B",
            "qwen2.5_32B_Instruct",
        ],
        "color": "#1f77b4",
        "title": "Qwen 2.5",
    },
}

# ---------------------------------------------------------------------------
# LOAD METRICS --------------------------------------------------------------
# ---------------------------------------------------------------------------
with JSON_PATH.open() as f:
    raw = json.load(f)

# ---------------------------------------------------------------------------
# PLOT ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharey=True)

err_cfg = dict(ecolor="black", capsize=3, capthick=0.6, linewidth=0.8)  # NEW

for ax, (family_key, info) in zip(axs, FAMILIES.items()):
    base_color = info["color"]
    variants   = info["variants"]

    labels: List[str]       = []
    det_vals: List[float]   = []
    nd_vals: List[float]    = []
    det_up_list:  List[float]    = []
    nd_up_list:   List[float]    = []
    det_down:  List[float]    = []
    nd_down:   List[float]    = []    

    for var in variants:
        exps = raw.get(var, {})
        # inside the variant loop
        det_m, nd_m, det_dn, det_up, nd_dn, nd_up = gather_metrics(exps)

        det_vals.append(det_m)
        nd_vals.append(nd_m)
        det_down.append(det_dn)
        det_up_list.append(det_up)        # renamed to avoid clash with function
        nd_down.append(nd_dn)
        nd_up_list.append(nd_up)

        # label: drop family prefix (lower-case) & prettify "-Instruct"
        fam_prefix = family_key.lower()
        labels.append(
            var.replace(f"{fam_prefix}_", "")
               .replace("-", "-\n")  # allow wrap for long labels
        )

    x      = np.arange(len(labels), dtype=np.float32)
    width  = 0.35

    det_colors    = [base_color] * len(variants)
    ndet_colors   = [lighten_color(c, 0.7) for c in det_colors]

    # build (2, N) yerr arrays: first row = down err (all zeros), second = up
    # build (2, N) arrays: [down; up]
    det_yerr = np.vstack([det_down, det_up_list])
    nd_yerr  = np.vstack([nd_down,  nd_up_list])


    print(det_yerr, nd_yerr)
    ax.bar(
        x - width / 2,
        det_vals,
        width,
        label="Unambiguous" if ax is axs[0] else None,
        color=det_colors,
        edgecolor="black",
        linewidth=0.4,
        yerr=det_yerr,          # NEW
        error_kw=err_cfg,       # NEW
    )
    ax.bar(
        x + width / 2,
        nd_vals,
        width,
        label="Ambiguous" if ax is axs[0] else None,  # NEW
        color=ndet_colors,
        edgecolor="black",
        linewidth=0.4,
        yerr=nd_yerr,           # NEW
        error_kw=err_cfg,       # NEW
    )

    ax.set_title(info["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0.9, 1.01)
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylabel("Accuracy")

fig.subplots_adjust(hspace=0.15)

handles, labels_legend = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels_legend,
    ncol=2,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
)

fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("visualisations/loremipsum.pdf")
