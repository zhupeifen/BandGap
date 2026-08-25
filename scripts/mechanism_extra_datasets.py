"""
Extra validation datasets for the solid-solution-share -> random-split-optimism
relationship (Section 3.4, Figure 4). These are NOT part of the six-dataset main
study; they are additional public band-gap datasets used only to test whether the
solid-solution-share vs. optimism trend holds beyond the original points.

For each dataset we report:
  - solid-solution share  = fraction of formulas with non-integer (reduced) stoichiometry
  - optimism              = by-chemistry non-zero MAE / random non-zero MAE
                            (two-stage if zero-inflated, else single-stage log1p regressor)

    .venv/Scripts/python.exe scripts/mechanism_extra_datasets.py
"""

import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from matminer.datasets import load_dataset
from matminer.utils.io import load_dataframe_from_json
from pymatgen.core import Composition

from cv_evaluation import magpie, metrics, DATA_DIR, NONMETAL_THRESHOLD, N_SPLITS, SEED

warnings.filterwarnings("ignore")


def frac_share(formulas):
    n = fr = 0
    for f in formulas:
        try:
            c = Composition(str(f))
        except Exception:
            continue
        n += 1
        a = np.array(list(c.get_el_amt_dict().values())); a = a / a.min()
        if not np.all(np.abs(a - np.round(a)) < 1e-3):
            fr += 1
    return fr / max(n, 1)


def chem_groups(formulas):
    def key(f):
        try:
            return "-".join(sorted(e.symbol for e in Composition(str(f)).elements))
        except Exception:
            return str(f)
    return np.array([key(f) for f in formulas])


def two_stage(X, y, tr, te):
    ybin = (y > 0).astype(int)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(ybin[tr] == 0).sum() / max((ybin[tr] == 1).sum(), 1),
        eval_metric="logloss", random_state=SEED)
    clf.fit(X.iloc[tr], ybin[tr])
    nz = tr[y[tr] > 0]
    reg = xgb.XGBRegressor(
        n_estimators=700, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.5, random_state=SEED)
    reg.fit(X.iloc[nz], np.log1p(y[nz]))
    gap_hat = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
    is_nm = clf.predict_proba(X.iloc[te])[:, 1] >= NONMETAL_THRESHOLD
    return metrics(is_nm * gap_hat, y[te])["nz_mae"]


def single_stage(X, y, tr, te):
    reg = xgb.XGBRegressor(
        n_estimators=700, max_depth=6, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.5, random_state=SEED)
    reg.fit(X.iloc[tr], np.log1p(y[tr]))
    pred = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
    return metrics(pred, y[te])["nz_mae"]


def optimism(X, y, groups, fit):
    rand = [fit(X, y, tr, te)
            for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X)]
    chem = [fit(X, y, tr, te)
            for tr, te in GroupKFold(N_SPLITS).split(X, y, groups)]
    r, c = float(np.mean(rand)), float(np.mean(chem))
    return r, c, c / r


def run(name, formulas, y):
    y = np.asarray(y, dtype=float)
    keep = ~np.isnan(y)
    formulas = list(np.asarray(formulas, dtype=object)[keep]); y = y[keep]
    X = magpie(formulas).astype(float).reset_index(drop=True)
    groups = chem_groups(formulas)
    zero_frac = float(np.mean(y == 0))
    fit = two_stage if zero_frac > 0.02 else single_stage
    fs = frac_share(formulas)
    r, c, opt = optimism(X, y, groups, fit)
    mode = "two-stage" if fit is two_stage else "single-stage"
    print(f"{name:<22}n={len(y):>6}  zeros={zero_frac:>5.1%}  %frac={fs:>6.1%}  "
          f"[{mode}]  randMAE={r:.3f}  chemMAE={c:.3f}  optimism={opt:.3f}")
    return name, fs, opt


if __name__ == "__main__":
    print("Additional validation datasets (mechanism check, not the main six):\n")
    results = []

    k = load_dataframe_from_json(str(DATA_DIR / "expt_gap_kingsbury.json"))
    results.append(run("expt_gap_kingsbury", k["formula"], k["expt_gap"]))

    d = load_dataset("dielectric_constant")
    results.append(run("dielectric (Petousis)", d["formula"], d["band_gap"]))

    print("\n(name, solid-solution share, optimism):")
    for nm, fs, opt in results:
        print(f"  {nm:<22} {fs:.4f}  {opt:.4f}")
