"""
Double-perovskite dataset: single-stage regression + grouped-split generalization.

This dataset (1,306 A2BB'O6 double perovskites, GLLB-SC gaps) has NO zero gaps, so
the two-stage zero-gap correction does not apply (as with the Tol set). Its role is
to extend the random-vs-grouped split generalization test (Section 5.4) to a fifth
dataset. We report single-stage regression MAE under random, by-composition, and
by-chemistry 5-fold splits.

    .venv/Scripts/python.exe scripts/double_perovskite_analysis.py
"""

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from pymatgen.core import Composition

from cv_evaluation import num, magpie, DATA_DIR, N_SPLITS, SEED

warnings.filterwarnings("ignore")

df = pd.read_csv(DATA_DIR / "Dataset_double_perovskites_gap_v1.csv")
y = num(df["gap gllbsc"]).to_numpy()
X = magpie(df["formula"]).astype(float)
print(f"rows: {len(df)}   zero gaps: {int((y == 0).sum())}   "
      f"gap range {y.min():.2f}-{y.max():.2f} eV")


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


g_comp = df["formula"].map(comp_key).to_numpy()
g_chem = df["formula"].map(chem_key).to_numpy()


def mae_over(splits):
    out = []
    for tr, te in splits:
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED)
        m.fit(X.iloc[tr], y[tr])
        out.append(np.mean(np.abs(m.predict(X.iloc[te]) - y[te])))
    return np.array(out)


strategies = [
    ("random", "-", list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X))),
    ("by-composition", np.unique(g_comp).size, list(GroupKFold(N_SPLITS).split(X, y, g_comp))),
    ("by-chemistry", np.unique(g_chem).size, list(GroupKFold(N_SPLITS).split(X, y, g_chem))),
]

print(f"\n=== Double perovskites (single-stage, {N_SPLITS}-fold, overall MAE eV) ===")
print(f"{'split':<16}{'n_groups':>10}{'MAE':>16}")
rand_mae = None
for label, ngrp, splits in strategies:
    maes = mae_over(splits)
    if rand_mae is None:
        rand_mae = maes.mean()
    print(f"{label:<16}{str(ngrp):>10}{maes.mean():>10.3f} +/-{maes.std():<5.3f}"
          f"   ({maes.mean()/rand_mae:.2f}x random)")
