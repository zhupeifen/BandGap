"""
Generalization stress-test for the Tol-screened perovskite band-gap model.

A plain random 80/20 split gives acc(<0.07 eV) = 0.994 / R2 = 0.9998. The dataset
is a dense grid (16,979 compositions x 4 crystal structures, every gap non-zero),
so a random split can place near-identical compositions in both train and test.
This script re-evaluates under progressively harder, leakage-controlled splits:

  1. random            - reference (no grouping)
  2. by-composition    - all 4 structure variants of a composition stay together
  3. by-chemistry      - whole element-set families held out (extrapolation)

If accuracy survives, the model generalizes; if it collapses, 0.994 was interpolation.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM = 46
TOL = 0.07

df = pd.read_csv(DATA_DIR / "Tol_screened_ensemble_final.csv")

site_cols = ["K", "Rb", "Cs", "MA", "FA",          # A-site
             "Ca", "Sr", "Ba", "Ge", "Sn", "Pb",   # B-site
             "Cl", "Br", "I"]                       # X-site
feat_cols = site_cols + [
    "A_ion_rad", "A_BP", "A_MP", "A_dens", "A_at_wt", "A_EA", "A_IE", "A_hof",
    "A_hov", "A_En", "A_at_num", "A_period",
    "B_ion_rad", "B_BP", "B_MP", "B_dens", "B_at_wt", "B_EA", "B_IE", "B_hof",
    "B_hov", "B_En", "B_at_num", "B_period",
    "X_ion_rad", "X_BP", "X_MP", "X_dens", "X_at_wt", "X_EA", "X_IE", "X_hof",
    "X_hov", "X_En", "X_at_num", "X_period",
    "t", "o", "tao", "Cubic", "Tetra", "Ortho", "Hex"]

X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
y = pd.to_numeric(df["Band gap(HSE-mf1)"], errors="coerce").to_numpy()

# Group labels -------------------------------------------------------------
# by-composition: rounded fraction vector over the 14 site columns
comp_round = df[site_cols].apply(pd.to_numeric, errors="coerce").round(4)
group_comp = comp_round.astype(str).agg("|".join, axis=1).to_numpy()

# by-chemistry: which elements are present at each site (ignore amounts)
present = (df[site_cols].apply(pd.to_numeric, errors="coerce") > 0)
group_chem = present.astype(int).astype(str).agg("".join, axis=1).to_numpy()

print(f"rows: {len(df)}   unique compositions: {pd.unique(group_comp).size}   "
      f"unique chemistries: {pd.unique(group_chem).size}")


def evaluate(tr, te):
    m = xgb.XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=RANDOM,
                         n_jobs=-1)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    err = np.abs(p - y[te])
    r2 = 1 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
    return np.mean(err < TOL), err.mean(), np.sqrt((err ** 2).mean()), r2


def split_groups(groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM)
    return next(gss.split(X, y, groups))


print(f"\n{'split':<16}{'n_test':>8}{'acc<0.07':>11}{'MAE':>9}{'RMSE':>9}{'R2':>9}")
# 1. random
tr, te = next(ShuffleSplit(1, test_size=0.2, random_state=RANDOM).split(X))
a, mae, rmse, r2 = evaluate(tr, te)
print(f"{'random':<16}{len(te):>8}{a:>11.4f}{mae:>9.4f}{rmse:>9.4f}{r2:>9.4f}")
# 2. by-composition
tr, te = split_groups(group_comp)
a, mae, rmse, r2 = evaluate(tr, te)
print(f"{'by-composition':<16}{len(te):>8}{a:>11.4f}{mae:>9.4f}{rmse:>9.4f}{r2:>9.4f}")
# 3. by-chemistry
tr, te = split_groups(group_chem)
a, mae, rmse, r2 = evaluate(tr, te)
print(f"{'by-chemistry':<16}{len(te):>8}{a:>11.4f}{mae:>9.4f}{rmse:>9.4f}{r2:>9.4f}")
