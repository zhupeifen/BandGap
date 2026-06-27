"""
Grouped-split generalization test for the two zero-inflated datasets.

Section 5.4 shows that random splits overstate generalization on the dense Tol
surrogate dataset. This script checks whether the same effect holds for the
*non-zero* metrics on Wolverton oxides and Castelli perovskites, by evaluating the
two-stage model under three 5-fold partitions:

  - random        : shuffled KFold (reference)
  - by-composition: GroupKFold on the reduced formula (polymorph twins kept together)
  - by-chemistry  : GroupKFold on the element set (whole chemistries held out)

    .venv/Scripts/python.exe scripts/cv_grouped_splits.py
"""

import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from matminer.utils.io import load_dataframe_from_json
from pymatgen.core import Composition

from cv_evaluation import (load_wolverton, load_castelli, load_expt_gap, metrics,
                           DATA_DIR, NONMETAL_THRESHOLD, N_SPLITS, SEED)

warnings.filterwarnings("ignore")


def groups_for(filename):
    df = load_dataframe_from_json(str(DATA_DIR / filename)).reset_index(drop=True)

    def comp_key(f):
        try:
            return Composition(str(f)).reduced_formula
        except Exception:
            return str(f)

    def chem_key(f):
        try:
            return "-".join(sorted(e.symbol for e in Composition(str(f)).elements))
        except Exception:
            return str(f)

    return df["formula"].map(comp_key).to_numpy(), df["formula"].map(chem_key).to_numpy()


def two_stage(X, y, tr, te):
    ybin = (y > 0).astype(int)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(ybin[tr] == 0).sum() / max((ybin[tr] == 1).sum(), 1),
        eval_metric="logloss", random_state=SEED)
    clf.fit(X.iloc[tr], ybin[tr])
    nz = tr[y[tr] > 0]
    reg = xgb.XGBRegressor(
        n_estimators=700, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_lambda=1.5, random_state=SEED)
    reg.fit(X.iloc[nz], np.log1p(y[nz]))
    gap_hat = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
    is_nm = clf.predict_proba(X.iloc[te])[:, 1] >= NONMETAL_THRESHOLD
    return metrics(is_nm * gap_hat, y[te])


def evaluate(X, y, splits):
    rows = [two_stage(X, y, tr, te) for tr, te in splits]
    def agg(key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std()
    return agg("nz_mae"), agg("nz_acc_030"), agg("overall_acc")


def run(name, X, y, filename):
    g_comp, g_chem = groups_for(filename)
    strategies = [
        ("random", "-", list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X))),
        ("by-composition", np.unique(g_comp).size,
         list(GroupKFold(N_SPLITS).split(X, y, g_comp))),
        ("by-chemistry", np.unique(g_chem).size,
         list(GroupKFold(N_SPLITS).split(X, y, g_chem))),
    ]
    print(f"\n=== {name}  (two-stage, {N_SPLITS}-fold, mean +/- std) ===")
    print(f"{'split':<16}{'n_groups':>10}{'non-zero MAE':>18}"
          f"{'non-zero acc<0.30':>20}{'overall acc':>16}")
    for label, ngrp, splits in strategies:
        (mae_m, mae_s), (a30_m, a30_s), (ov_m, ov_s) = evaluate(X, y, splits)
        print(f"{label:<16}{str(ngrp):>10}"
              f"{mae_m:>10.3f} +/-{mae_s:<5.3f}"
              f"{a30_m:>11.3f} +/-{a30_s:<5.3f}"
              f"{ov_m:>8.3f} +/-{ov_s:<5.3f}")


if __name__ == "__main__":
    name, _, X_full, y, _ = load_wolverton()
    run(name, X_full, y, "wolverton_oxides.json")
    name, _, X_full, y, _ = load_castelli()
    run(name, X_full, y, "castelli_perovskites.json")
    name, _, X_full, y, _ = load_expt_gap()
    run(name, X_full, y, "expt_gap.json")
