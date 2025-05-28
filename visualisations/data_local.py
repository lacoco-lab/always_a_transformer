# ------------------------------
# plot_tasks.py  (single file: data + plotter)
# ------------------------------
"""
Usage::
    python plot_tasks.py

Produces a 10‑subplot figure (2 × 5) – one panel per task – showing the
three‑bin accuracies.  **Copying** tasks are drawn with circles, **retrieval**
tasks with squares.  Colours reflect expressiveness (green ✔ vs red ✖).
A compact legend is centred *slightly* above the figure.  Output is
written to ``visualisations/tasks_accuracy.pdf`` (auto‑creating the
folder if necessary).
"""

from pathlib import Path
from typing import Dict
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ------------------------------
# Data – accuracy numbers per task
# ------------------------------

tasks: Dict[str, Dict[str, float]] = {
    # ――――――――――――― Copying tasks ―――――――――――――
    "UF": {
        "name": "UF (Unique Forward)",
        "Bin 0": 1.0,
        "Bin 1": 0.9945,
        "Bin 2": 0.97,
        "Type": "copying",
        "Expressive": True,
    },
    "UB": {
        "name": "UB (Unique Backward)",
        "Bin 0": 1.0,
        "Bin 1": 0.99,
        "Bin 2": 0.9724,
        "Type": "copying",
        "Expressive": True,
    },
    "NF": {
        "name": "NF",
        "Bin 0": 1.0,
        "Bin 1": 0.4675,
        "Bin 2": 0.1286,
        "Type": "copying",
        "Expressive": False,
    },
    "NB": {
        "name": "NB",
        "Bin 0": 0.9995,
        "Bin 1": 0.2680,
        "Bin 2": 0.0512,
        "Type": "copying",
        "Expressive": False,
    },

    # ――――――――――――― Retrieval tasks ――――――――――――
    "UR": {
        "name": "UR",
        "Bin 0": 1.0,
        "Bin 1": 0.9915,
        "Bin 2": 0.99,
        "Type": "retrieval",
        "Expressive": True,
    },
    "UL": {
        "name": "UL",
        "Bin 0": 1.0,
        "Bin 1": 0.99,
        "Bin 2": 0.99,
        "Type": "retrieval",
        "Expressive": True,
    },
    "NRfirst": {
        "name": "NRFirst",
        "Bin 0": 1.0,
        "Bin 1": 1.0,
        "Bin 2": 0.9995,
        "Type": "retrieval",
        "Expressive": True,
    },
    "NLfirst": {
        "name": "NLFirst",
        "Bin 0": 1.0,
        "Bin 1": 1.0,
        "Bin 2": 0.99,
        "Type": "retrieval",
        "Expressive": True,
    },   
    "NRLast": {
        "name": "NRLast",
        "Bin 0": 1.0,
        "Bin 1": 0.8605,
        "Bin 2": 0.4685,
        "Type": "retrieval",
        "Expressive": False,
    },     
    "NLLast": {
        "name": "NLLast",
        "Bin 0": 1.0,
        "Bin 1": 0.7700,
        "Bin 2": 0.4720,
        "Type": "retrieval",
        "Expressive": False,
    },
}

# ------------------------------
# Plotting utilities
# ------------------------------

# Global style – bigger fonts, bold everywhere
mpl.rcParams.update({
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

# Marker & colour maps
MARKER = {"copying": "o", "retrieval": "s"}         # circle / square
COLORS = {True: "seagreen", False: "crimson"}           # expressive / not

BINS  = ["Bin 0", "Bin 1", "Bin 2"]
XPOSS = np.arange(len(BINS)) + 1  # 1‑based x‑positions

# Layout: 2 × 5 grid for 10 tasks
N_TASKS = len(tasks)
COLS     = 5
ROWS     = math.ceil(N_TASKS / COLS)
FIGSIZE  = (COLS * 3.8, ROWS * 3.8)

# ------------------------------
# Main plotting function
# ------------------------------

def plot_tasks(savepath: str = "visualisations/tasks_accuracy.pdf") -> None:
    # Ensure output directory exists
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(ROWS, COLS, figsize=FIGSIZE, constrained_layout=True)
    axes      = axes.flatten()

    # One mini‑plot per task
    for idx, (tid, info) in enumerate(tasks.items()):
        ax    = axes[idx]
        yvals = [100 * info[b] for b in BINS]  # convert to %

        ax.plot(
            XPOSS,
            yvals,
            marker     = MARKER[info["Type"]],
            linestyle   = "-",
            linewidth   = 2,
            markersize  = 10,
            color       = COLORS[info["Expressive"]],
        )

        ax.set_xticks(XPOSS)
        ax.set_xticklabels(["Bin 1", "Bin 2", "Bin 3"])
        ax.set_ylim(-5, 105)
        ax.set_title(info["name"], fontsize=mpl.rcParams["font.size"])
        ax.grid(True, alpha=0.4)

    # Hide any unused axis (if task count changes)
    for extra_ax in axes[len(tasks):]:
        extra_ax.axis("off")

    # Legend – Copying vs Retrieval, expressive colour cue in lines
    legend_handles = [
        mlines.Line2D([], [], marker="o", linestyle="-", color="green", label="In C-RASP[Pos]"),
        mlines.Line2D([], [], marker="o", linestyle="-", color="red", label="Not in C-RASP[Pos]"),
    ]
    fig.legend(
        handles       = legend_handles,
        loc           = "upper center",
        ncol          = 2,
        frameon       = False,
        bbox_to_anchor= (0.5, 1.1),  # ← little above the top border
        fontsize      = mpl.rcParams["font.size"] + 2,
    )
    plt.savefig(savepath, bbox_inches="tight")
    print(f"Saved → {savepath}")


# ------------------------------
# CLI entry point
# ------------------------------

if __name__ == "__main__":
    plot_tasks()
