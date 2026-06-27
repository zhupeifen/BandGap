"""
Figure: MAE under leakage-controlled splits, relative to the random split.

Visualizes the dataset-dependence of random-split optimism (Section 5.4). Numbers
are the cross-validated MAE values from tol_generalization_test.py (overall MAE,
the Tol target has no zeros) and cv_grouped_splits.py (non-zero MAE). Each dataset
is normalized to its own random-split MAE so the three sit on one axis.

    .venv/Scripts/python.exe scripts/make_generalization_figure.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# MAE (eV) per split strategy: [random, by-composition, by-chemistry]
DATA = {
    "Tol perovskites\n(overall MAE)": [0.0148, 0.0176, 0.0242],
    "expt_gap (experimental)\n(non-zero MAE)": [0.292, 0.435, 0.454],
    "MP gap (106k, PBE)\n(non-zero MAE)": [0.550, 0.622, 0.628],
    "Double perovskites\n(overall MAE)": [0.246, 0.255, 0.248],
    "Wolverton oxides\n(non-zero MAE)": [0.334, 0.326, 0.340],
    "Castelli perovskites\n(non-zero MAE)": [0.585, 0.570, 0.586],
}
SPLITS = ["random", "by-composition", "by-chemistry"]

fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(SPLITS))
width = 0.13
colors = ["#4C72B0", "#C44E52", "#937860", "#8172B3", "#DD8452", "#55A868"]

n = len(DATA)
for i, (name, maes) in enumerate(DATA.items()):
    rel = np.array(maes) / maes[0]                 # relative to random split
    bars = ax.bar(x + (i - (n - 1) / 2) * width, rel, width, label=name, color=colors[i])
    for b, r, m in zip(bars, rel, maes):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=7)

ax.axhline(1.0, color="gray", lw=1, ls="--", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(SPLITS)
ax.set_ylabel("MAE relative to random split")
ax.set_title("Random-split optimism scales with compositional redundancy\n"
             "(large: Tol, expt_gap; moderate: MP gap; "
             "negligible: low-redundancy sets)", fontsize=11)
ax.set_ylim(0.9, 1.8)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "plots"
fig.savefig(out / "generalization_splits.svg")
fig.savefig(out / "generalization_splits.png", dpi=150)
print(f"Wrote {out / 'generalization_splits.png'}")
