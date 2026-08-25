"""
Robustness of the solid-solution-share vs. optimism rank correlation (Spearman 0.94).

The correlation is over a small set of datasets, so we quantify its stability three ways:
  - exact permutation test  : all n! rank permutations -> an exact p-value (no large-n
                              approximation, appropriate for this small n);
  - jackknife (leave-one-dataset-out) : recompute Spearman dropping each dataset;
  - bootstrap 95% CI        : resample datasets with replacement (reported with the
                              caveat that bootstrap CIs for a correlation are wide and
                              only indicative at this n).

Pairs are recomputed from the datasets (solid-solution share via the same routine as
redundancy_correlation.py); mp_gap is excluded (≈0% fractional, as in the main analysis).

    .venv/Scripts/python.exe scripts/mechanism_bootstrap.py
"""

from itertools import permutations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from matminer.utils.io import load_dataframe_from_json

from redundancy_correlation import stats_from_formulas, OPTIMISM, EXTRA, DATA_DIR

warnings.filterwarnings("ignore")
N_BOOT = 20000


def build_pairs():
    pairs = []
    for name, fn in [("Wolverton", "wolverton_oxides.json"),
                     ("Castelli", "castelli_perovskites.json"),
                     ("expt_gap", "expt_gap.json")]:
        df = load_dataframe_from_json(str(DATA_DIR / fn))
        _, frac, _, _ = stats_from_formulas(df["formula"])
        pairs.append((name, frac, OPTIMISM[name]))
    dp = pd.read_csv(DATA_DIR / "Dataset_double_perovskites_gap_v1.csv")
    _, frac, _, _ = stats_from_formulas(dp["formula"])
    pairs.append(("Double perov", frac, OPTIMISM["Double perov"]))
    # Tol: fractional by construction (>3 occupied sites = mixed/solid-solution)
    tol = pd.read_csv(DATA_DIR / "Tol_screened_ensemble_final.csv")
    site = ["K", "Rb", "Cs", "MA", "FA", "Ca", "Sr", "Ba", "Ge", "Sn", "Pb", "Cl", "Br", "I"]
    nsite = (tol[site].apply(pd.to_numeric, errors="coerce") > 0).sum(axis=1)
    pairs.append(("Tol", float((nsite > 3).mean()), OPTIMISM["Tol"]))
    for name, (fr, o) in EXTRA.items():
        pairs.append((name, fr, o))
    return pairs


def exact_perm_p(x, y):
    rho0 = stats.spearmanr(x, y)[0]
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    count = tot = 0
    for perm in permutations(range(len(ry))):
        r = stats.spearmanr(rx, ry[list(perm)])[0]
        tot += 1
        if abs(r) >= abs(rho0) - 1e-12:
            count += 1
    return rho0, count / tot


def main():
    pairs = build_pairs()
    names = [p[0] for p in pairs]
    x = np.array([p[1] for p in pairs])
    y = np.array([p[2] for p in pairs])
    n = len(x)
    print(f"n = {n} datasets (mp_gap excluded, ~0% fractional)")
    for nm, fr, o in pairs:
        print(f"  {nm:<14} solid-solution share = {fr:6.1%}   optimism = {o:.3f}")

    rho, p_perm = exact_perm_p(x, y)
    pr = stats.pearsonr(x, y)
    print(f"\nSpearman rho = {rho:.3f}   exact permutation p = {p_perm:.4f}")
    print(f"Pearson  r   = {pr[0]:.3f}   (p = {pr[1]:.4f})")

    # jackknife
    jk = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        jk.append(stats.spearmanr(x[m], y[m])[0])
    jk = np.array(jk)
    print(f"\nJackknife (leave-one-dataset-out) Spearman: "
          f"min {jk.min():.3f}, max {jk.max():.3f}")
    print("  most influential drop: "
          + ", ".join(f"-{names[i]}:{jk[i]:.2f}" for i in np.argsort(jk)[:2]))

    # bootstrap
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(stats.spearmanr(x[idx], y[idx])[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nBootstrap 95% CI (n={len(boots)} valid resamples): [{lo:.2f}, {hi:.2f}]  "
          f"(wide at this n; indicative only)")
    print(f"Bootstrap median rho = {np.median(boots):.3f}")

    import json
    (DATA_DIR / "mechanism_bootstrap_results.json").write_text(json.dumps(
        {"n": n, "pairs": [(nm, fr, o) for nm, fr, o in pairs],
         "spearman": float(rho), "perm_p": float(p_perm),
         "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
         "jackknife_min": float(jk.min()), "jackknife_max": float(jk.max()),
         "bootstrap_ci": [float(lo), float(hi)]}, indent=2))


if __name__ == "__main__":
    main()
