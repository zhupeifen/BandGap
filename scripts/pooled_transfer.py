"""
Pooled multi-fidelity and stacked-transfer experiment (R2 revision, Reviewer 1 point 5,
Reviewer 2 point 4).

Question: can data from other datasets / levels of theory lower the chemistry-grouped
(extrapolation) error on a target dataset?  Tree models cannot be fine-tuned, so we test
the two tree-compatible forms of transfer, on the common composition-only Magpie space,
with the non-zero regressor only (no gate, so the effect is not confounded by gating):

  baseline    : Magpie-only XGBoost regressor trained on the target's own train folds
  +MP feature : a regressor trained on the 59,962 non-zero MP (PBE) materials supplies its
                prediction as one extra feature (stacked transfer)
  pooled+ind  : one regressor trained on the union of the non-zero rows of all five
                composition-representable datasets, with a one-hot dataset/fidelity
                indicator (multi-fidelity pooling)
  pooled-ind  : the same union with NO indicator (negative control: label heterogeneity)

Evaluation: chemistry-grouped 5-fold GroupKFold on each target (expt, Wolverton, Castelli),
identical folds across conditions.  Composition-disjoint source: rows of any *other*
dataset whose reduced formula occurs in the target's test fold are dropped from the
training pool for that fold, so no test composition is seen at any fidelity.

    .venv/Scripts/python.exe scripts/pooled_transfer.py
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from matminer.utils.io import load_dataframe_from_json
from pymatgen.core import Composition

from cv_evaluation import magpie, num, DATA_DIR, SEED, N_SPLITS

warnings.filterwarnings("ignore")

REG = dict(n_estimators=700, max_depth=6, learning_rate=0.03, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.5, random_state=SEED)
NAMES = ["expt", "wolverton", "castelli", "double", "mp"]
OUT = DATA_DIR / "pooled_transfer_results.json"


def keys(formulas):
    red, chem = [], []
    for f in formulas:
        try:
            c = Composition(str(f))
            red.append(c.reduced_formula)
            chem.append("-".join(sorted(e.symbol for e in c.elements)))
        except Exception:
            red.append(str(f)); chem.append(str(f))
    return np.array(red), np.array(chem)


def load():
    """dict name -> (X Magpie DataFrame, y, redform, chem), non-zero rows only."""
    out = {}
    specs = [("expt", "expt_gap.json", "gap expt"),
             ("wolverton", "wolverton_oxides.json", "gap pbe"),
             ("castelli", "castelli_perovskites.json", "gap gllbsc")]
    for name, fn, col in specs:
        df = load_dataframe_from_json(str(DATA_DIR / fn)).reset_index(drop=True)
        y = num(df[col]).to_numpy(float); nz = y > 0
        X = magpie(df["formula"][nz]).astype(float).reset_index(drop=True)
        red, chem = keys(df["formula"][nz])
        out[name] = (X, y[nz], red, chem)
    dp = pd.read_csv(DATA_DIR / "Dataset_double_perovskites_gap_v1.csv")
    y = num(dp["gap gllbsc"]).to_numpy(float); nz = y > 0
    X = magpie(dp["formula"][nz]).astype(float).reset_index(drop=True)
    red, chem = keys(dp["formula"][nz])
    out["double"] = (X, y[nz], red, chem)
    d = pd.read_pickle(DATA_DIR / "_mp_gap_features.pkl")
    y = np.asarray(d["y"], float); nz = y > 0
    X = d["X"][nz].reset_index(drop=True).astype(float)
    X.columns = out["expt"][0].columns  # same 132 Magpie labels, enforce identical order
    out["mp"] = (X, y[nz], np.asarray(d["redform"])[nz], np.asarray(d["chem"])[nz])
    for k, (X, y, r, c) in out.items():
        print(f"  {k:<10} n_nonzero={len(y):>6}  chemistries={len(set(c)):>6}")
    return out


def fit(X, y):
    m = xgb.XGBRegressor(**REG); m.fit(X, np.log1p(y)); return m


def pred(m, X):
    return np.clip(np.expm1(m.predict(X)), 0, None)


def with_indicator(X, name):
    X = X.copy()
    for n in NAMES:
        X[f"src_{n}"] = float(n == name)
    return X


def run_target(data, target):
    X, y, red, chem = data[target]
    others = [n for n in NAMES if n != target]
    folds = list(GroupKFold(N_SPLITS).split(X, y, chem))
    res = {c: [] for c in ["baseline", "mp_feature", "pooled_ind", "pooled_noind"]}
    dropped = []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        test_red = set(red[te])
        # --- baseline
        res["baseline"].append(np.mean(np.abs(pred(fit(X.iloc[tr], y[tr]), X.iloc[te]) - y[te])))
        # --- stacked transfer: MP source model, composition-disjoint from the test fold
        Xm, ym, rm, _ = data["mp"]
        keep = ~np.isin(rm, list(test_red))
        m_src = fit(Xm[keep], ym[keep])
        Xs = X.copy(); Xs["mp_pred"] = pred(m_src, X)
        res["mp_feature"].append(np.mean(np.abs(pred(fit(Xs.iloc[tr], y[tr]), Xs.iloc[te]) - y[te])))
        # --- pooled training (with / without indicator)
        parts_X, parts_y, ndrop = [with_indicator(X.iloc[tr], target)], [y[tr]], {}
        for n in others:
            Xo, yo, ro, _ = data[n]
            keep = ~np.isin(ro, list(test_red))
            ndrop[n] = int((~keep).sum())
            parts_X.append(with_indicator(Xo[keep], n)); parts_y.append(yo[keep])
        Xp = pd.concat(parts_X, ignore_index=True); yp = np.concatenate(parts_y)
        Xte = with_indicator(X.iloc[te], target)
        res["pooled_ind"].append(np.mean(np.abs(pred(fit(Xp, yp), Xte) - y[te])))
        ind = [c for c in Xp.columns if c.startswith("src_")]
        res["pooled_noind"].append(np.mean(np.abs(
            pred(fit(Xp.drop(columns=ind), yp), Xte.drop(columns=ind)) - y[te])))
        dropped.append(ndrop)
        print(f"    fold {k}: " + "  ".join(f"{c}={v[-1]:.3f}" for c, v in res.items())
              + f"  (pool n={len(yp)}, dropped={ndrop}, {time.time()-t0:.0f}s)")
    summary = {c: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": [float(x) for x in v]}
               for c, v in res.items()}
    summary["dropped_overlap_per_fold"] = dropped
    return summary


def main():
    print("loading / featurizing ...")
    data = load()
    results = {}
    for target in ["expt", "wolverton", "castelli"]:
        print(f"\n=== target: {target} (chemistry-grouped {N_SPLITS}-fold, non-zero regressor, Magpie) ===")
        results[target] = run_target(data, target)
        OUT.write_text(json.dumps(results, indent=2))
    print("\n=== summary: non-zero MAE (eV), mean +/- std over grouped folds ===")
    print(f"{'target':<10}{'baseline':>16}{'+MP feature':>16}{'pooled+ind':>16}{'pooled-ind':>16}")
    for t, r in results.items():
        print(f"{t:<10}" + "".join(f"{r[c]['mean']:>10.3f}±{r[c]['std']:<5.3f}"
                                   for c in ["baseline", "mp_feature", "pooled_ind", "pooled_noind"]))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
