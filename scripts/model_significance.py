"""
Is XGBoost's lead over other regressors statistically real, or within noise?

Repeated 5-fold CV (10 seeds x 5 folds = 50 paired fold-scores) on the non-zero subset,
identical splits across models so the per-fold scores are paired. We compare every model
to XGBoost with (i) the Nadeau-Bengio corrected resampled t-test -- which inflates the
naive variance by (1/J + rho) to account for the overlapping training sets that make a
plain paired t-test anti-conservative -- and (ii) the Wilcoxon signed-rank test as a
distribution-free secondary check. rho = n_test/n_train = 1/(k-1) for k-fold CV.

    .venv/Scripts/python.exe scripts/model_significance.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from cv_evaluation import load_wolverton, load_castelli, load_expt_gap
from model_comparison import xgb_model, cat_model, lgbm_model, mlp_model

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEEDS = list(range(10))
K = 5
RHO = 1.0 / (K - 1)                      # n_test / n_train for k-fold
MODELS = [("XGBoost", xgb_model, False), ("CatBoost", cat_model, False),
          ("LightGBM", lgbm_model, False), ("MLP (Magpie)", mlp_model, True)]


def fold_mae(make, Xtr, Xte, ytr, yte, impute):
    if impute:
        med = Xtr.median()
        Xtr = Xtr.fillna(med).fillna(0.0)
        Xte = Xte.fillna(med).fillna(0.0)
    m = make()
    m.fit(Xtr, np.log1p(ytr))
    pred = np.clip(np.expm1(np.clip(m.predict(Xte), None, 6.0)), 0, None)
    return float(np.mean(np.abs(pred - yte)))


def corrected_t(diff):
    """Nadeau-Bengio corrected resampled paired t-test. diff = scoreA - scoreB per fold."""
    d = np.asarray(diff, float)
    J = len(d)
    mean = d.mean()
    var = d.var(ddof=1)
    if var == 0:
        return mean, (0.0 if mean == 0 else np.inf), 1.0 if mean == 0 else 0.0
    t = mean / np.sqrt((1.0 / J + RHO) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return mean, t, p


def main():
    out = {}
    for loader in (load_wolverton, load_castelli, load_expt_gap):
        name, _xb, X_full, y, _c = loader()
        X = X_full.astype(float)
        nz = y > 0
        Xn = X[nz].reset_index(drop=True)
        yn = np.asarray(y, float)[nz]
        print(f"\n=== {name}  (n_nonzero={nz.sum()}, {len(SEEDS)}x{K} repeated CV) ===")

        per_fold = {m[0]: [] for m in MODELS}
        for seed in SEEDS:
            for tr, te in KFold(K, shuffle=True, random_state=seed).split(Xn):
                Xtr, Xte = Xn.iloc[tr], Xn.iloc[te]
                ytr, yte = yn[tr], yn[te]
                for mname, make, impute in MODELS:
                    per_fold[mname].append(fold_mae(make, Xtr.copy(), Xte.copy(),
                                                    ytr, yte, impute))

        ref = np.array(per_fold["XGBoost"])
        rows = {}
        print(f"{'model':<16}{'MAE (mean+/-std)':>20}{'dMAE vs XGB':>14}"
              f"{'p (corr-t)':>12}{'p (Wilcoxon)':>14}")
        for mname, _, _ in MODELS:
            v = np.array(per_fold[mname])
            mu, sd = v.mean(), v.std()
            if mname == "XGBoost":
                print(f"{mname:<16}{f'{mu:.3f}+/-{sd:.3f}':>20}{'-- (ref)':>14}{'':>12}{'':>14}")
                rows[mname] = dict(mae_mean=float(mu), mae_std=float(sd),
                                   per_fold=v.tolist())
                continue
            diff = v - ref                      # model - XGB (positive => worse than XGB)
            dmean, _t, p_t = corrected_t(diff)
            try:
                p_w = stats.wilcoxon(v, ref).pvalue
            except ValueError:
                p_w = 1.0
            print(f"{mname:<16}{f'{mu:.3f}+/-{sd:.3f}':>20}{f'{dmean:+.3f}':>14}"
                  f"{p_t:>12.3f}{p_w:>14.3f}")
            rows[mname] = dict(mae_mean=float(mu), mae_std=float(sd),
                               dmae_vs_xgb=float(dmean), p_corrected_t=float(p_t),
                               p_wilcoxon=float(p_w), per_fold=v.tolist())
        out[name] = dict(n_nonzero=int(nz.sum()), models=rows)

    (DATA_DIR / "model_significance_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {DATA_DIR / 'model_significance_results.json'}")


if __name__ == "__main__":
    main()
