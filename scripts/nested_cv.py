"""
Nested cross-validation with leak-free model/threshold selection + significance.

Addresses two rigor gaps in cv_evaluation.py:
  - the gating threshold tau and the regressor hyperparameters were fixed / chosen
    with knowledge of the test split;
  - improvements were reported with error bars but no significance test.

Here, for each OUTER fold, an INNER holdout (carved from the outer-train fold only)
selects the regressor config and tau by minimizing OVERALL MAE on the inner-val set
(no contact with the outer-test fold). The selected model is refit on the full
outer-train fold and scored on the outer-test fold. We then run a paired test across
outer folds comparing single-stage baseline vs. two-stage non-zero MAE.

    .venv/Scripts/python.exe scripts/nested_cv.py
"""

import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy import stats

from cv_evaluation import load_wolverton, load_castelli, metrics, SEED

warnings.filterwarnings("ignore")

OUTER = 10            # outer folds (evaluation + paired significance)
TOL = 0.07
REG_GRID = [
    dict(max_depth=5, learning_rate=0.03, min_child_weight=2, reg_lambda=1.5),
    dict(max_depth=6, learning_rate=0.03, min_child_weight=2, reg_lambda=1.5),
    dict(max_depth=7, learning_rate=0.04, min_child_weight=1, reg_lambda=1.0),
]
TAU_GRID = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15]


def fit_classifier(X, y):
    ybin = (y > 0).astype(int)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(ybin == 0).sum() / max((ybin == 1).sum(), 1),
        eval_metric="logloss", random_state=SEED)
    clf.fit(X, ybin)
    return clf


def fit_regressor(X, y, cfg):
    nz = y > 0
    reg = xgb.XGBRegressor(n_estimators=700, subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, **cfg)
    reg.fit(X[nz], np.log1p(y[nz]))
    return reg


def select(Xtr, ytr):
    """Inner holdout selection of (regressor cfg, tau) by overall MAE."""
    idx = np.arange(len(ytr))
    itr, ival = next(KFold(4, shuffle=True, random_state=SEED).split(idx))  # 25% inner-val
    clf = fit_classifier(Xtr.iloc[itr], ytr[itr])
    proba_val = clf.predict_proba(Xtr.iloc[ival])[:, 1]
    yval = ytr[ival]
    best, best_obj = None, np.inf
    for cfg in REG_GRID:
        reg = fit_regressor(Xtr.iloc[itr], ytr[itr], cfg)
        gap_val = np.clip(np.expm1(reg.predict(Xtr.iloc[ival])), 0, None)
        for tau in TAU_GRID:
            pred = (proba_val >= tau) * gap_val
            obj = np.mean(np.abs(pred - yval))          # overall MAE on inner-val
            if obj < best_obj:
                best_obj, best = obj, (cfg, tau)
    return best


def run(name, X, y, base_cfg):
    outer = KFold(OUTER, shuffle=True, random_state=SEED)
    base_rows, two_rows, taus, depths = [], [], [], []
    for tr, te in outer.split(X):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]

        bm = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, **base_cfg)
        bm.fit(Xtr, ytr)
        base_rows.append(metrics(bm.predict(Xte), yte))

        cfg, tau = select(Xtr, ytr)
        taus.append(tau); depths.append(cfg["max_depth"])
        clf = fit_classifier(Xtr, ytr)
        reg = fit_regressor(Xtr, ytr, cfg)
        gap_hat = np.clip(np.expm1(reg.predict(Xte)), 0, None)
        is_nm = clf.predict_proba(Xte)[:, 1] >= tau
        two_rows.append(metrics(is_nm * gap_hat, yte))

    def col(rows, key):
        return np.array([r[key] for r in rows])

    print(f"\n=== {name}  (nested {OUTER}-fold CV, mean +/- std) ===")
    print(f"{'metric':<22}{'baseline':>18}{'two-stage (nested)':>22}")
    for key, label in [("nz_mae", "non-zero MAE (eV)"),
                       ("nz_rmse", "non-zero RMSE (eV)"),
                       ("nz_acc_007", "non-zero acc<0.07"),
                       ("nz_acc_030", "non-zero acc<0.30"),
                       ("overall_acc", "overall acc<0.07")]:
        b, t = col(base_rows, key), col(two_rows, key)
        print(f"{label:<22}{b.mean():>8.3f} +/-{b.std():<6.3f}"
              f"{t.mean():>10.3f} +/-{t.std():<6.3f}")

    b_mae, t_mae = col(base_rows, "nz_mae"), col(two_rows, "nz_mae")
    diff = b_mae - t_mae
    t_p = stats.ttest_rel(b_mae, t_mae).pvalue
    try:
        w_p = stats.wilcoxon(b_mae, t_mae).pvalue
    except ValueError:
        w_p = float("nan")
    print(f"\n  non-zero MAE reduction: {diff.mean():.3f} eV "
          f"({100 * diff.mean() / b_mae.mean():.1f}%), improved in "
          f"{int((diff > 0).sum())}/{OUTER} folds")
    print(f"  paired t-test p = {t_p:.2e} | Wilcoxon p = {w_p:.2e}")
    from collections import Counter
    print(f"  selected tau (per fold): {dict(Counter(taus))} | "
          f"selected max_depth: {dict(Counter(depths))}")


if __name__ == "__main__":
    for loader in (load_wolverton, load_castelli):
        name, _X_base, X_full, y, base_cfg = loader()
        run(name, X_full, y, base_cfg)
