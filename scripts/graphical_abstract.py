"""
Graphical abstract: the paper's two core messages in one wide graphic.
  (a) The metric illusion -- high aggregate accuracy obscures large non-zero-gap error,
      which the non-zero metric + two-stage model expose and recover.
  (b) Exploratory split sensitivity versus fractional-formula share.

All numbers are the manuscript's reported values (hardcoded; no computation).

    .venv/Scripts/python.exe scripts/graphical_abstract.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "plots"
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})

GREEN, RED, BLUE, PURPLE = "#2ca02c", "#d62728", "#1f77b4", "#7a3b9a"

# (b) source-formula fractional share vs grouped/random MAE ratio
SS = np.array([0.0, 0.0, 8.6, 0.0, 99.8, 0.0])
OPT = np.array([1.018, 1.002, 1.555, 1.008, 1.635, 1.139])
NAMES = ["Wolverton", "Castelli", "Expt gap", "Double\nperov.", "Tol", "Dielectric"]


def main():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.patch.set_alpha(0)

    # ---- (a) metric illusion (Castelli single-stage) ----
    axl.patch.set_alpha(0)
    axl.bar([0], [0.93], width=0.6, color=GREEN, zorder=3)
    axl.set_ylim(0, 1.05); axl.set_ylabel("Aggregate accuracy", color=GREEN)
    axl.tick_params(axis="y", colors=GREEN)
    axl.text(0, 0.94, "0.93", ha="center", va="bottom", color=GREEN, fontsize=15, fontweight="bold")

    ax2 = axl.twinx()
    ax2.bar([1], [1.0], width=0.6, color=RED, zorder=3)
    ax2.bar([2], [0.585], width=0.6, color="#f0a3a3", zorder=3)
    ax2.set_ylim(0, 1.25); ax2.set_ylabel("Non-zero MAE (eV)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.text(1, 1.01, "~1.0", ha="center", va="bottom", color=RED, fontsize=15, fontweight="bold")
    ax2.text(2, 0.66, "0.585", ha="center", va="bottom", color="#c0504d", fontsize=13, fontweight="bold")

    axl.set_xticks([0, 1, 2])
    axl.set_xticklabels(["aggregate\naccuracy", "single-stage\nnon-zero MAE", "two-stage\nnon-zero MAE"])
    axl.set_xlim(-0.6, 2.6)
    axl.set_title("(a) Aggregate accuracy obscures non-zero-gap error")
    # Land the arrow on the bar's left shoulder: aimed at the bar top it collided with the 0.585 label.
    ax2.annotate("", xy=(1.78, 0.60), xytext=(1, 1.02),
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6, shrinkB=2,
                                 connectionstyle="arc3,rad=-0.35"))
    ax2.text(1.62, 0.80, "two-stage\ncorrection", ha="center", va="center",
             fontsize=11, color="0.35")

    # ---- (b) mechanism ----
    axr.patch.set_alpha(0)
    axr.scatter(SS, OPT, s=90, color=PURPLE, zorder=3, edgecolor="white", linewidth=0.8)
    # label the three separated datasets individually
    for name, s, o, dx, ha in [("Tol", 99.8, 1.635, -4, "right"),
                               ("Expt gap", 8.6, 1.555, 4, "left"),
                               ("Dielectric", 0.0, 1.139, 5, "left")]:
        axr.annotate(name, (s, o), xytext=(s + dx, o + 0.015), fontsize=12, ha=ha,
                     va="bottom", color="0.25")
    # one grouped label for the three stoichiometric datasets clustered near (0, 1.0)
    axr.annotate("Wolverton, Castelli,\nDouble perovskite", xy=(0.6, 1.005),
                 xytext=(22, 1.05), fontsize=12, ha="left", va="center", color="0.25",
                 arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    axr.axhline(1.0, color="k", lw=0.8, alpha=0.4, ls="--")
    axr.set_xlim(-6, 116); axr.set_ylim(0.95, 1.72)
    axr.set_xlabel("Fractional-formula share (%)")
    axr.set_ylabel("Grouped/random MAE ratio")
    axr.set_title("(b) Fractional formulas flag split sensitivity")
    axr.text(55, 1.10, r"Spearman $\rho = 0.85$", fontsize=14, color=PURPLE, fontweight="bold")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "graphical_abstract.png", dpi=600, transparent=False,
                facecolor="white", bbox_inches="tight")
    # vector copy: stays sharp at any zoom (line art + text)
    fig.savefig(OUT / "graphical_abstract.pdf", transparent=False,
                facecolor="white", bbox_inches="tight")
    print(f"Wrote {OUT / 'graphical_abstract.png'} (600 dpi) and .pdf (vector)")


if __name__ == "__main__":
    main()
