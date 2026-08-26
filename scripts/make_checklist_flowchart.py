"""
Reporting checklist as a flowchart (manuscript Figure 7): the eight items of Table 12 in the order a
practitioner meets them, each with the quantity to report. Reviewer 1 asked for "a concise table or
flowchart"; the table carries the failure mode each step guards against and where this study measures
it, and this figure carries the sequence. The failure mode each step guards against stays in the table, and
the caption says so, so the two are not redundant.

Same drawing conventions as Figure 1 (scripts/make_schematic.py): Times New Roman, transparent
background, rounded boxes, equal aspect so a tight crop cannot distort it.

    .venv/Scripts/python.exe scripts/make_checklist_flowchart.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

OUT = Path(__file__).resolve().parent.parent / "plots"

C_DATA = "#dce6f1"     # what the data are
C_PROT = "#d6ecd8"     # how it is split and attributed
C_GEN = "#fce1cc"      # how far it generalizes
C_REP = "#e9dcf2"      # how it is reported
EDGE = "#2f2f2f"
ARR = "#3a3a3a"
HEAD = "#ededed"

# (title, what to do, what to report, colour)
STEPS = [
    ("1. Zero-gap treatment", "Classify metals, then regress on the\nnon-zero materials, and score the two apart.",
     "non-zero fraction and operational zero;\ngate recall and AUC", C_DATA),
    ("2. Evaluation metrics", "Score the true non-zero subset in eV.",
     "non-zero MAE and RMSE, beside any\naggregate tolerance metric", C_DATA),
    ("3. Split strategy", "Split at random, by composition, and by\nchemistry when both regimes matter.",
     "grouped/random MAE ratio; fractional-\nformula share; grouping rule", C_PROT),
    ("4. Attributing a gain", "Hold the features fixed and select every\nconfiguration inside the training folds.",
     "feature-held-fixed difference;\nuncertainty or p-value; protocol", C_PROT),
    ("5. Transfer tests", "Predict across datasets, not only within one.",
     "transfer matrix; MAE with and\nwithout the offset correction", C_GEN),
    ("6. Uncertainty", "Repeat over folds or seeds, deep models\nincluded.",
     "mean ± SD over at least five folds\nor seeds; interval coverage", C_GEN),
    ("7. Level of theory", "Read every error against the label it was\ntrained on.",
     "functional or experimental source;\ncross-dataset offset", C_REP),
    ("8. Reproducibility", "Archive the code that produced the numbers.",
     "repository tag and version DOI", C_REP),
]

BW, BH, GAP = 58.0, 19.0, 4.5      # box width, box height, vertical gap
COLX = (32.0, 98.0)                # column centres
TOP = 78.0                         # centre of the first box in each column
HDR_Y = 95.0                       # centre of the column headers
RET_X = 65.0                       # the return connector runs up the gap between the columns


def box(ax, cx, cy, title, action, report, fc):
    ax.add_patch(FancyBboxPatch((cx - BW / 2, cy - BH / 2), BW, BH,
                 boxstyle="round,pad=0.5,rounding_size=2.0", linewidth=1.4,
                 edgecolor=EDGE, facecolor=fc, zorder=2))
    ax.text(cx - BW / 2 + 2.5, cy + BH / 2 - 3.2, title, ha="left", va="center",
            fontsize=13, fontweight="bold", zorder=3)
    ax.text(cx - BW / 2 + 2.5, cy + 1.0, action, ha="left", va="center",
            fontsize=11.5, linespacing=1.35, zorder=3)
    ax.text(cx - BW / 2 + 2.5, cy - BH / 2 + 4.2, "Report: " + report, ha="left", va="center",
            fontsize=11, linespacing=1.35, style="italic", color="#333333", zorder=3)


def arrow(ax, cx, y_from, y_to):
    ax.annotate("", xy=(cx, y_to), xytext=(cx, y_from), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=ARR, lw=1.9,
                                shrinkA=0, shrinkB=0, mutation_scale=18))


def header(ax, cx, y, text):
    ax.add_patch(FancyBboxPatch((cx - BW / 2, y - 4.2), BW, 8.4,
                 boxstyle="round,pad=0.4,rounding_size=1.5", linewidth=1.1,
                 edgecolor="#6a6a6a", facecolor=HEAD, zorder=2))
    ax.text(cx, y, text, ha="center", va="center", fontsize=12.5, fontweight="bold", zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(11, 9.3))
    ax.set_aspect("equal")
    ax.set_xlim(0, 130); ax.set_ylim(-8, 102); ax.axis("off")

    header(ax, COLX[0], HDR_Y, "Set up the comparison")
    header(ax, COLX[1], HDR_Y, "Qualify it, then report it")

    for i, (title, action, report, fc) in enumerate(STEPS):
        cx = COLX[i // 4]
        cy = TOP - (i % 4) * (BH + GAP)
        box(ax, cx, cy, title, action, report, fc)
        if i % 4:
            arrow(ax, cx, cy + BH / 2 + GAP - 0.5, cy + BH / 2 + 0.5)

    # step 4 wraps round to step 5
    last, first = TOP - 3 * (BH + GAP) - BH / 2, TOP
    ax.plot([COLX[0], COLX[0], RET_X, RET_X], [last - 0.5, last - 5.0, last - 5.0, first],
            color="#8a8a8a", lw=1.5, linestyle=(0, (5, 3)), zorder=1,
            solid_capstyle="round", solid_joinstyle="round")
    ax.annotate("", xy=(COLX[1] - BW / 2 - 0.5, first), xytext=(RET_X, first), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=1.5, linestyle=(0, (5, 3)),
                                shrinkA=0, shrinkB=0, mutation_scale=17))

    OUT.mkdir(exist_ok=True)
    path = OUT / "reporting_checklist_flowchart.png"
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
    print("Wrote", path)


if __name__ == "__main__":
    main()
