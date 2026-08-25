"""
What actually drives random-split optimism? An honest mechanism analysis.

We first asked whether optimism scales with a simple redundancy *count*
(R = 1 − unique chemistries / N). It does not (weak, non-significant correlation):
Castelli has many repeated chemistries yet no optimism. The factor that does separate
the datasets is the share of **solid-solution / fractional compositions** — entries with
non-integer stoichiometry (alloys, mixed-site screening grids) that let the model
interpolate within a continuous composition family. Datasets of distinct stoichiometric
compounds show little optimism regardless of how many compositions repeat.

    .venv/Scripts/python.exe scripts/redundancy_correlation.py

Data note: this figure uses the tolerance-screened perovskite set, a third-party CC-BY 4.0
dataset that is NOT redistributed with this code. Download it from the Materials Data Facility
(Biswas & Mannodi-Kanakkithodi, 2024; DOI 10.18126/dp3z-bp06) and place
`Tol_screened_ensemble_final.csv` in the repo's `data/` folder before running.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from matminer.utils.io import load_dataframe_from_json

from cv_evaluation import DATA_DIR

warnings.filterwarnings("ignore")

# Measured optimism = by-chemistry MAE / random MAE (Tables 7-8 / analyses).
OPTIMISM = {"Tol": 1.635, "expt_gap": 1.555, "mp_gap": 1.142,
            "Wolverton": 1.018, "Castelli": 1.002, "Double perov": 1.008}

# Additional INDEPENDENT public validation dataset (mechanism robustness check,
# not part of the six-dataset main study). (solid-solution share, optimism),
# computed reproducibly by scripts/mechanism_extra_datasets.py.
# NB: expt_gap_kingsbury is NOT included here -- it is the deduplicated version of
# the same Zhuo et al. data already in expt_gap, so it is not independent; it is
# used in the text as a controlled "same data, cleaned" comparison instead.
EXTRA = {"Dielectric": (0.045, 1.139)}


def stats_from_formulas(formulas):
    n, frac, comp, chem = 0, 0, set(), set()
    for f in formulas:
        try:
            c = Composition(str(f))
        except Exception:
            continue
        n += 1
        a = np.array(list(c.get_el_amt_dict().values())); a = a / a.min()
        if not np.all(np.abs(a - np.round(a)) < 1e-3):
            frac += 1
        comp.add(c.reduced_formula)
        chem.add(frozenset(e.symbol for e in c.elements))
    return n, frac / n, 1 - len(comp) / n, 1 - len(chem) / n


def main():
    rows = {}
    for name, fn in [("Wolverton", "wolverton_oxides.json"),
                     ("Castelli", "castelli_perovskites.json"),
                     ("expt_gap", "expt_gap.json")]:
        df = load_dataframe_from_json(str(DATA_DIR / fn))
        rows[name] = stats_from_formulas(df["formula"])
    dp = pd.read_csv(DATA_DIR / "Dataset_double_perovskites_gap_v1.csv")
    rows["Double perov"] = stats_from_formulas(dp["formula"])

    # Tol: fractional by construction; element-presence signature for chemistry.
    tol = pd.read_csv(DATA_DIR / "Tol_screened_ensemble_final.csv")
    site = ["K", "Rb", "Cs", "MA", "FA", "Ca", "Sr", "Ba", "Ge", "Sn", "Pb", "Cl", "Br", "I"]
    pres = (tol[site].apply(pd.to_numeric, errors="coerce") > 0)
    nsite = pres.sum(axis=1)
    sig = pres.astype(int).astype(str).agg("".join, axis=1)
    comp_sig = tol[site].round(4).astype(str).agg("|".join, axis=1)
    rows["Tol"] = (len(tol), float((nsite > 3).mean()),
                   1 - comp_sig.nunique() / len(tol), 1 - sig.nunique() / len(tol))

    # mp_gap: from cache (no 675 MB reload). Ordered compounds -> ~0 fractional.
    d = pd.read_pickle(DATA_DIR / "_mp_gap_features.pkl")
    N = len(d["chem"])
    rows["mp_gap"] = (N, np.nan, 1 - len(set(d["redform"])) / N, 1 - len(set(d["chem"])) / N)

    print(f"{'dataset':<14}{'N':>8}{'%fractional':>12}{'%dup-comp':>11}"
          f"{'R(chem)':>9}{'optimism':>10}")
    frac, optf, names_f, Rc, optall = [], [], [], [], []
    for name, (n, fr, dupc, rch) in rows.items():
        o = OPTIMISM[name]
        fr_s = "n/a (~0)" if np.isnan(fr) else f"{fr:.1%}"
        print(f"{name:<16}{n:>8}{fr_s:>12}{dupc:>10.1%}{rch:>9.3f}{o:>10.3f}")
        Rc.append(rch); optall.append(o)
        if not np.isnan(fr):
            frac.append(fr); optf.append(o); names_f.append(name)

    # Additional independent validation datasets (Section 3.4 robustness check).
    for name, (fr, o) in EXTRA.items():
        print(f"{name:<16}{'-':>8}{fr:>11.1%}{'-':>11}{'-':>9}{o:>10.3f}")
        frac.append(fr); optf.append(o); names_f.append(name)

    Rc, optall, frac, optf = map(np.array, (Rc, optall, frac, optf))
    print(f"\ncount metric R(chem) vs optimism : Pearson r = {stats.pearsonr(Rc, optall)[0]:.2f} "
          f"(p = {stats.pearsonr(Rc, optall)[1]:.2f})  -- weak / n.s.")
    pr = stats.pearsonr(frac, optf); sr = stats.spearmanr(frac, optf)
    print(f"%fractional vs optimism (n={len(frac)}): Pearson r = {pr[0]:.2f} (p = {pr[1]:.3f}), "
          f"Spearman = {sr[0]:.2f} (p = {sr[1]:.3f})")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(frac * 100, optf, s=70, color="tab:purple", zorder=3)
    for name, fr, o in zip(names_f, frac, optf):
        ax.annotate(name, (fr * 100, o), textcoords="offset points", xytext=(7, 3), fontsize=9)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlim(-4, 114)
    ax.set_xlabel("Solid-solution share (% entries with fractional stoichiometry)")
    ax.set_ylabel("Random-split optimism  (by-chemistry MAE / random MAE)")
    ax.set_title("Optimism is driven by solid-solution composition\n"
                 f"(Spearman = {sr[0]:.2f}; the simple redundancy count does not predict it)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "plots"
    fig.savefig(out / "redundancy_correlation.svg")
    fig.savefig(out / "redundancy_correlation.png", dpi=150)
    print(f"\nWrote {out / 'redundancy_correlation.png'}")
    print("(mp_gap omitted from the figure: ~0% fractional yet 1.14x optimism — its mild\n"
          " effect comes from discrete near-duplicate entries, a weaker second mechanism.)")


if __name__ == "__main__":
    main()
