"""
Two small analyses requested in review (R2):

1. Robustness of the fractional-share vs optimism correlation to leaving out the Tol surrogate
   set and/or the Petousis set (exact permutation p-values, Spearman and Pearson).
2. Chemistry overlap between the four transfer datasets (element-set Jaccard similarity and the
   share of test chemistries whose element set occurs in the training set), and its relation to
   the bias-corrected transfer MAE from data/transfer_matrix_results.json.

    .venv/Scripts/python.exe scripts/transfer_chemistry_overlap.py
"""
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
try:
    from matminer.utils.io import load_dataframe_from_json
except ImportError:
    def load_dataframe_from_json(path):
        return pd.read_json(path, orient="split")
from pymatgen.core import Composition

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def num(values):
    return pd.to_numeric(values, errors="coerce")

warnings.filterwarnings("ignore")
OUT = DATA_DIR / "transfer_chemistry_overlap_results.json"

# fractional share (%) and random-split optimism, from redundancy_correlation.py (2026-08-25 run)
SETS = {"Tol": (99.8, 1.635), "expt": (8.6, 1.555), "Petousis": (0.0, 1.139),
        "Wolverton": (0.0, 1.018), "Castelli": (0.0, 1.002), "Double": (0.0, 1.008)}


def exact_perm_p(x, y, stat):
    """Exact two-sided permutation p for a correlation statistic (n <= 7)."""
    obs = stat(x, y)
    y = np.asarray(y); k = 0; n = 0
    for perm in itertools.permutations(range(len(y))):
        n += 1
        if abs(stat(x, y[list(perm)])) >= abs(obs) - 1e-12:
            k += 1
    return obs, k / n


def correlation_variants():
    res = {}
    for label, drop in [("all six", []), ("without Tol", ["Tol"]), ("without Petousis", ["Petousis"]),
                        ("without Tol and Petousis", ["Tol", "Petousis"])]:
        names = [k for k in SETS if k not in drop]
        x = np.array([SETS[k][0] for k in names]); y = np.array([SETS[k][1] for k in names])
        rho, p_rho = exact_perm_p(x, y, lambda a, b: stats.spearmanr(a, b)[0])
        r, p_r = exact_perm_p(x, y, lambda a, b: stats.pearsonr(a, b)[0])
        res[label] = {"n": len(names), "datasets": names, "spearman": float(rho), "p_spearman_exact": p_rho,
                      "pearson": float(r), "p_pearson_exact": p_r}
        print(f"{label:<26} n={len(names)}  Spearman {rho:.2f} (exact p={p_rho:.3f})  Pearson {r:.2f} (p={p_r:.3f})")
    return res


def chem_sets(formulas):
    out = []
    for f in formulas:
        try:
            out.append(frozenset(e.symbol for e in Composition(str(f)).elements))
        except Exception:
            pass
    return out


def transfer_overlap():
    specs = [("Expt", "expt_gap.json", "gap expt"), ("Wolverton", "wolverton_oxides.json", "gap pbe"),
             ("Castelli", "castelli_perovskites.json", "gap gllbsc")]
    chem, elems = {}, {}
    for name, fn, col in specs:
        df = load_dataframe_from_json(str(DATA_DIR / fn)).reset_index(drop=True)
        y = num(df[col]).to_numpy(float)
        cs = chem_sets(df["formula"][y > 0])
        chem[name] = cs; elems[name] = set().union(*cs)
    dp = pd.read_csv(DATA_DIR / "Dataset_double_perovskites_gap_v1.csv")
    cs = chem_sets(dp["formula"][num(dp["gap gllbsc"]).to_numpy(float) > 0])
    chem["Double"] = cs; elems["Double"] = set().union(*cs)
    tm = json.loads((DATA_DIR / "transfer_matrix_results.json").read_text())
    labels = ["Expt", "Wolverton", "Castelli", "Double"]
    mae = np.array(tm["mae"]); bc = np.array(tm["mae_bias_corrected"]); rr = np.array(tm["pearson_r"])
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            jac = len(elems[a] & elems[b]) / len(elems[a] | elems[b])
            train_chems = set(chem[a])
            coverage = np.mean([c in train_chems for c in chem[b]])          # test chemistries seen in training
            excess = bc[i, j] - mae[j, j]                                    # corrected transfer MAE minus target's own OOF MAE
            pairs.append({"train": a, "test": b, "mae": float(mae[i, j]), "mae_bias_corrected": float(bc[i, j]),
                          "pearson_r": float(rr[i, j]), "within_target_mae": float(mae[j, j]),
                          "excess_over_within": float(excess), "element_jaccard": float(jac),
                          "test_chemistry_coverage": float(coverage)})
    df = pd.DataFrame(pairs).sort_values("mae")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cov = df["test_chemistry_coverage"].to_numpy(); jac = df["element_jaccard"].to_numpy()
    for name, col in [("bias-corrected MAE", df["mae_bias_corrected"]), ("excess over within-target MAE", df["excess_over_within"]),
                      ("pearson r", df["pearson_r"])]:
        v = col.to_numpy()
        print(f"Spearman({name}, chemistry coverage) = {stats.spearmanr(v, cov)[0]:+.2f} (p={stats.spearmanr(v, cov)[1]:.3f});  "
              f"vs element Jaccard = {stats.spearmanr(v, jac)[0]:+.2f} (p={stats.spearmanr(v, jac)[1]:.3f})")
    return pairs


def main():
    res = {"correlation_variants": correlation_variants(), "transfer_pairs": transfer_overlap()}
    OUT.write_text(json.dumps(res, indent=2))
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
