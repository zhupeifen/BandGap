"""
Pipeline schematic (manuscript Figure 1): the two-stage classify-then-regress
band-gap model with threshold gating, plus the evaluation protocol. Rendered in
Times New Roman on a transparent background, natively at 4:3. The connectors are
elbow (right-angle) paths so the branch (featurize -> classifier & regressor) and
the merge (both -> gate) read clearly.

    .venv/Scripts/python.exe scripts/make_schematic.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

OUT = Path(__file__).resolve().parent.parent / "plots"

C_IN = "#dce6f1"      # input / features
C_CLF = "#d6ecd8"     # classifier
C_REG = "#fce1cc"     # regressor
C_GATE = "#e9dcf2"    # gate
C_OUT = "#ededed"     # output / note
EDGE = "#2f2f2f"
ARR = "#3a3a3a"

# 4:3 data canvas (x range : y range = 4:3) so the figure renders undistorted.
XMAX, YMAX = 130, 97.5


def box(ax, cx, cy, w, h, title, body, fc, fs=15):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.5,rounding_size=2.2", linewidth=1.4,
                 edgecolor=EDGE, facecolor=fc, zorder=2))
    ax.text(cx, cy + h / 2 - 4.0, title, ha="center", va="center",
            fontsize=fs + 1, fontweight="bold", zorder=3)
    ax.text(cx, cy - 2.2, body, ha="center", va="center", fontsize=fs - 1, zorder=3)


def line(ax, pts, lw=1.9):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ARR, lw=lw, zorder=1,
            solid_capstyle="round", solid_joinstyle="round")


def head(ax, p1, p2, lw=1.9):
    ax.annotate("", xy=p2, xytext=p1, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=ARR, lw=lw, shrinkA=0, shrinkB=1.5,
                                mutation_scale=20))


def elbow(ax, pts):
    """Right-angle polyline with an arrowhead on the final segment."""
    line(ax, pts)
    head(ax, pts[-2], pts[-1])


def main():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_aspect("equal")               # equal units -> no distortion under a tight crop
    ax.set_xlim(0, 146); ax.set_ylim(8, 92); ax.axis("off")

    box(ax, 13, 60, 20, 22, "Material", "formula +\nnative columns", C_IN)
    box(ax, 41, 60, 20, 22, "Featurize", "Magpie (132) +\nnative columns", C_IN)
    box(ax, 78, 80, 26, 18, "Classifier", "metal / non-metal\n→ P(non-metal)", C_CLF)
    box(ax, 78, 40, 26, 18, "Regressor", "non-zero gaps\nlog(1 + gap)", C_REG)
    box(ax, 112, 60, 18, 21, "Gate", "[P ≥ τ] × ĝap\nτ = 0.25", C_GATE)
    box(ax, 133, 60, 16, 22, "Predicted", "band gap\n(0 if metal)", C_OUT, fs=14)

    # input chain
    elbow(ax, [(23, 60), (31, 60)])                           # material -> featurize
    # branch: featurize -> classifier (up) and regressor (down)
    line(ax, [(51, 60), (58, 60)])                            # shared stem
    elbow(ax, [(58, 60), (58, 80), (65, 80)])                 # -> classifier
    elbow(ax, [(58, 60), (58, 40), (65, 40)])                 # -> regressor
    # merge: classifier & regressor -> gate (Y-join, single clear arrow in)
    line(ax, [(91, 80), (98, 80), (98, 60)])                  # classifier -> merge node
    line(ax, [(91, 40), (98, 40), (98, 60)])                  # regressor  -> merge node
    elbow(ax, [(98, 60), (103, 60)])                          # merge -> gate
    # output
    elbow(ax, [(121, 60), (125, 60)])                         # gate -> predicted gap

    note = ("Evaluation: report non-zero-subset MAE / RMSE (not aggregate accuracy),\n"
            "under random / by-composition / by-chemistry splits")
    ax.add_patch(FancyBboxPatch((6, 13), 134, 11, boxstyle="round,pad=0.5,rounding_size=1.5",
                 linewidth=1.1, edgecolor="#6a6a6a", facecolor=C_OUT, zorder=2))
    ax.text(73, 18.5, note, ha="center", va="center", fontsize=12.5, zorder=3)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "pipeline_schematic.png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {OUT / 'pipeline_schematic.png'}")


if __name__ == "__main__":
    main()
