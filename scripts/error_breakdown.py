"""
Where does the gap regressor fail? Physical error analysis.

A single aggregate MAE hides systematic failure modes. Using out-of-fold two-stage
predictions (5-fold CV), we break the non-zero regression error |gap_hat - gap| down
by (a) gap magnitude and (b) chemical family, revealing where the achievable accuracy
is set by physics/data rather than by the model.

    .venv/Scripts/python.exe scripts/error_breakdown.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
from matminer.utils.io import load_dataframe_from_json
from pymatgen.core import Composition

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import load_wolverton, load_castelli, load_expt_gap, DATA_DIR
from gate_analysis import out_of_fold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "plots"

GAP_BINS = [0, 1, 2, 3, 4, 99]
GAP_LABELS = ["0-1", "1-2", "2-3", "3-4", ">4"]
HALIDE = {"F", "Cl", "Br", "I"}
CHALC = {"S", "Se", "Te"}
PNICT = {"N", "P", "As", "Sb"}
COLORS = {"Wolverton": "#1f77b4", "Castelli": "#d62728", "Expt gap": "#2ca02c"}

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 20,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})


def anion_class(formula):
    try:
        els = {e.symbol for e in Composition(str(formula)).elements}
    except Exception:
        return "other"
    if "O" in els:
        return "oxide"
    if els & HALIDE:
        return "halide"
    if els & CHALC:
        return "chalcogenide"
    if els & PNICT:
        return "pnictide"
    return "other"


def oof_errors(loader, jsonfile, gapcol):
    name, _xb, X_full, y, _c = loader()
    formulas = load_dataframe_from_json(str(DATA_DIR / jsonfile)).reset_index(drop=True)["formula"]
    p, gap_hat, _ = out_of_fold(X_full.astype(float), y)
    nz = y > 0
    err = np.abs(gap_hat - y)
    return name, y[nz], err[nz], formulas[nz].to_numpy()


def main():
    specs = [(load_wolverton, "wolverton_oxides.json", "gap pbe", "Wolverton"),
             (load_castelli, "castelli_perovskites.json", "gap gllbsc", "Castelli"),
             (load_expt_gap, "expt_gap.json", "gap expt", "Expt gap")]
    data = {}
    for loader, jf, gc, short in specs:
        print(f"  {short}: out-of-fold two-stage ...")
        _n, yv, ev, fv = oof_errors(loader, jf, gc)
        data[short] = (yv, ev, fv)

    # (a) MAE by gap magnitude
    gap_table = {}
    for short, (yv, ev, _f) in data.items():
        idx = np.digitize(yv, GAP_BINS) - 1
        gap_table[short] = [float(ev[idx == b].mean()) if np.any(idx == b) else np.nan
                            for b in range(len(GAP_LABELS))]

    # (b) MAE by chemical family, on the diverse experimental set
    yv, ev, fv = data["Expt gap"]
    cls = np.array([anion_class(f) for f in fv])
    fam_order = ["oxide", "halide", "chalcogenide", "pnictide", "other"]
    fam_mae, fam_n = [], []
    for c in fam_order:
        m = cls == c
        fam_mae.append(float(ev[m].mean()) if m.any() else np.nan)
        fam_n.append(int(m.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax in axes:
        ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)

    ax = axes[0]
    shorts = list(data)
    x = np.arange(len(GAP_LABELS)); w = 0.26
    for i, short in enumerate(shorts):
        ax.bar(x + (i - 1) * w, gap_table[short], w, label=short, color=COLORS[short])
    ax.set_xticks(x); ax.set_xticklabels(GAP_LABELS)
    ax.set_xlabel("True gap (eV)"); ax.set_ylabel("Non-zero MAE (eV)")
    ax.set_title("(a) Error by gap magnitude")
    ax.legend(frameon=False)

    ax = axes[1]
    xf = np.arange(len(fam_order))
    ax.bar(xf, fam_mae, 0.6, color="#2ca02c")
    for xi, (mae, nn) in enumerate(zip(fam_mae, fam_n)):
        if not np.isnan(mae):
            ax.text(xi, mae, f"n={nn}", ha="center", va="bottom", fontsize=15)
    ax.set_xticks(xf); ax.set_xticklabels(fam_order, rotation=20)
    ax.set_ylabel("Non-zero MAE (eV)")
    ax.set_title("(b) Error by chemical family (Expt gap)")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "error_breakdown.png", dpi=300, transparent=True, bbox_inches="tight")
    (DATA_DIR / "error_breakdown_results.json").write_text(json.dumps(
        {"gap_bins": {"labels": GAP_LABELS, "mae": gap_table},
         "chemical_family_expt_gap": {"families": fam_order, "mae": fam_mae, "n": fam_n}},
        indent=2))
    print(f"\nWrote {OUT / 'error_breakdown.png'}")
    print("  gap-range MAE:", {k: [round(v, 2) for v in vv] for k, vv in gap_table.items()})
    print("  family MAE (expt_gap):", dict(zip(fam_order, [round(m, 2) for m in fam_mae])))


if __name__ == "__main__":
    main()
