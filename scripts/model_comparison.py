"""
Regressor comparison on the non-zero band-gap subset: can another model beat
XGBoost? Same features (Magpie + native columns), same 5-fold CV, same log1p
target. Reports non-zero MAE (mean +/- std) for XGBoost, CatBoost, LightGBM, and
TabPFN. (The two-stage gating is orthogonal; this isolates regressor accuracy.)

    .venv/Scripts/python.exe scripts/model_comparison.py
"""

import warnings
import time

import numpy as np
from sklearn.model_selection import KFold

import xgboost as xgb
from cv_evaluation import load_wolverton, load_castelli, load_expt_gap, N_SPLITS, SEED

warnings.filterwarnings("ignore")


def xgb_model():
    return xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                            reg_lambda=1.5, random_state=SEED)


def cat_model():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=700, depth=6, learning_rate=0.03,
                             l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
                             allow_writing_files=False)


def lgbm_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                         random_state=SEED, verbose=-1)


def tabpfn_model():
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)


def mlp_model():
    """Feed-forward neural net on the same Magpie features (standardized).
    Tests whether a non-tree learner breaks the gradient-boosting ceiling."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                     alpha=1e-3, learning_rate_init=1e-3, max_iter=1000,
                     early_stopping=True, n_iter_no_change=20, random_state=SEED))


MODELS = [("XGBoost", xgb_model, False), ("CatBoost", cat_model, False),
          ("LightGBM", lgbm_model, False), ("MLP (Magpie)", mlp_model, True)]
# TabPFN omitted: tabpfn>=2 requires a (free) Prior Labs account to fetch weights,
# which cannot be authenticated in a headless run. To include it, log in once with
# `python -c "from tabpfn import TabPFNRegressor; TabPFNRegressor()"` then add:
#   ("TabPFN", tabpfn_model, True)


def mae_cv(make_model, X, y, impute):
    """5-fold non-zero MAE for a regressor on log1p(gap). impute=True fills NaN
    with per-fold medians (TabPFN cannot take NaN); others use native NaN."""
    nz = y > 0
    Xn = X[nz].reset_index(drop=True)
    yn = y[nz]
    maes = []
    for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xn):
        Xtr, Xte = Xn.iloc[tr].copy(), Xn.iloc[te].copy()
        if impute:
            med = Xtr.median()
            Xtr = Xtr.fillna(med).fillna(0.0)
            Xte = Xte.fillna(med).fillna(0.0)
        m = make_model()
        m.fit(Xtr, np.log1p(yn[tr]))
        # Clip in log-space before expm1: a NN can occasionally emit a huge value
        # that overflows expm1 to inf. log1p(gap) <= ~3 for any real gap, so an
        # upper clip at 6 (expm1 ~ 400 eV) never touches a sensible prediction.
        pred = np.clip(np.expm1(np.clip(m.predict(Xte), None, 6.0)), 0, None)
        maes.append(np.mean(np.abs(pred - yn[te])))
    return float(np.mean(maes)), float(np.std(maes))


def main():
    datasets = []
    for loader in (load_wolverton, load_castelli, load_expt_gap):
        name, _xb, X_full, y, _c = loader()
        datasets.append((name, X_full.astype(float), y))

    print(f"Non-zero MAE (eV), {N_SPLITS}-fold CV, mean +/- std  "
          f"(same Magpie+native features, log1p target)\n")
    header = f"{'dataset':<26}" + "".join(f"{n:>18}" for n, _, _ in MODELS)
    print(header)
    for name, X, y in datasets:
        nz = int((y > 0).sum())
        cells = []
        for mname, make, impute in MODELS:
            try:
                t = time.time()
                mu, sd = mae_cv(make, X, y, impute)
                cells.append(f"{mu:.3f}+/-{sd:.3f}")
            except Exception as e:
                cells.append(f"FAILED:{type(e).__name__}")
        print(f"{name + f' (n={nz})':<26}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
