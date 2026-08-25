"""
Causal test of the solid-solution mechanism.

The cross-dataset correlation (solid-solution share vs random-split optimism,
Spearman 0.94; redundancy_correlation.py) is associational. Here we *intervene*:
on a single dataset (Tol), holding the target, features, and model fixed, we vary
only the DENSITY of solid-solution sampling -- the number of distinct fractional
compositions retained per chemistry family -- and measure how random-split optimism
responds. If optimism rises with solid-solution density and approaches 1 when each
family keeps a single composition (no within-family interpolation), the share is a
cause, not a correlate.

To isolate solid-solution interpolation we first deduplicate to one row per
composition, removing the structure-twin leakage channel; the only remaining
within-family leakage is fractional-neighbor interpolation.

    .venv/Scripts/python.exe scripts/mechanism_causal.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "plots"

K_VALUES = [1, 2, 4, 8, 16, 100]      # max distinct compositions kept per chemistry family
SEEDS = [0, 1, 2, 3]
TEST = 0.2

SITE = ["K", "Rb", "Cs", "MA", "FA", "Ca", "Sr", "Ba", "Ge", "Sn", "Pb", "Cl", "Br", "I"]
FEAT = SITE + [
    "A_ion_rad", "A_BP", "A_MP", "A_dens", "A_at_wt", "A_EA", "A_IE", "A_hof", "A_hov",
    "A_En", "A_at_num", "A_period",
    "B_ion_rad", "B_BP", "B_MP", "B_dens", "B_at_wt", "B_EA", "B_IE", "B_hof", "B_hov",
    "B_En", "B_at_num", "B_period",
    "X_ion_rad", "X_BP", "X_MP", "X_dens", "X_at_wt", "X_EA", "X_IE", "X_hof", "X_hov",
    "X_En", "X_at_num", "X_period",
    "t", "o", "tao", "Cubic", "Tetra", "Ortho", "Hex"]

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 20,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})


def load_tol_dedup():
    df = pd.read_csv(DATA_DIR / "Tol_screened_ensemble_final.csv")
    pres = (df[SITE].apply(pd.to_numeric, errors="coerce") > 0)
    chem = pres.astype(int).astype(str).agg("".join, axis=1)
    comp = df[SITE].apply(pd.to_numeric, errors="coerce").round(4).astype(str).agg("|".join, axis=1)
    df = df.assign(_chem=chem, _comp=comp).drop_duplicates("_comp").reset_index(drop=True)
    X = df[FEAT].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = pd.to_numeric(df["Band gap(HSE-mf1)"], errors="coerce").to_numpy()
    return X, y, df["_chem"].to_numpy(), df["_comp"].to_numpy()


def thin(chem, comp, k, rng):
    """Keep at most k distinct compositions per chemistry family."""
    d = pd.DataFrame({"chem": chem, "comp": comp, "i": np.arange(len(chem))})
    keep = []
    for _, g in d.groupby("chem"):
        comps = g["comp"].unique()
        sel = rng.choice(comps, size=min(k, len(comps)), replace=False)
        keep.append(g["i"].to_numpy()[np.isin(g["comp"].to_numpy(), sel)])
    return np.sort(np.concatenate(keep))


def mae_split(X, y, tr, te):
    m = xgb.XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1)
    m.fit(X[tr], y[tr])
    return float(np.mean(np.abs(m.predict(X[te]) - y[te])))


def main():
    cache = DATA_DIR / "mechanism_causal_results.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        print("Loaded cached causal results "
              "(delete data/mechanism_causal_results.json to recompute).")
    else:
        X, y, chem, comp = load_tol_dedup()
        print(f"Tol deduplicated to {len(y)} unique compositions; "
              f"{len(np.unique(chem))} chemistry families.")
        rows = []
        for k in K_VALUES:
            opt, dens, rnds, chems = [], [], [], []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                idx = thin(chem, comp, k, rng)
                Xk, yk, ck = X[idx], y[idx], chem[idx]
                dens.append(len(idx) / len(np.unique(ck)))      # mean comps per family
                tr, te = next(ShuffleSplit(1, test_size=TEST, random_state=s).split(Xk))
                r = mae_split(Xk, yk, tr, te)
                tr, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=s).split(Xk, yk, ck))
                c = mae_split(Xk, yk, tr, te)
                rnds.append(r); chems.append(c); opt.append(c / r)
            rows.append(dict(k=k, density=float(np.mean(dens)),
                             optimism=float(np.mean(opt)), optimism_std=float(np.std(opt)),
                             random_mae=float(np.mean(rnds)), bychem_mae=float(np.mean(chems))))
            print(f"  k={k:<4} density={rows[-1]['density']:.2f} comps/family  "
                  f"random={rows[-1]['random_mae']:.3f}  by-chem={rows[-1]['bychem_mae']:.3f}  "
                  f"optimism={rows[-1]['optimism']:.3f}±{rows[-1]['optimism_std']:.3f}")
        cache.write_text(json.dumps(rows, indent=2))

    dens = np.array([r["density"] for r in rows])
    opt = np.array([r["optimism"] for r in rows])
    err = np.array([r["optimism_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ax.errorbar(dens, opt, yerr=err, marker="o", ms=6, lw=1.8, capsize=3,
                color="tab:purple", zorder=3)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5, ls="--")
    ax.text(dens.max(), 1.0, " no optimism", va="bottom", ha="right", fontsize=15, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Solid-solution density (compositions / family)")
    ax.set_ylabel("Optimism (by-chem / random MAE)")
    ax.set_title("Increasing solid-solution density causes split optimism\n"
                 "(same dataset, target, features, and model)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "mechanism_causal.png", dpi=300, transparent=True, bbox_inches="tight")
    print(f"\nWrote {OUT / 'mechanism_causal.png'}")


if __name__ == "__main__":
    main()
