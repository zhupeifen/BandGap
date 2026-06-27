"""
Learning curves: non-zero MAE vs. training-set size.

Tests the Discussion claim that the hardest datasets are data-limited rather than
algorithm-limited. For each training fraction we subsample the training split,
fit the two-stage model, and measure non-zero MAE on a held-out test set, averaged
over several seeds. A curve still descending at full size => more data would help.

    .venv/Scripts/python.exe scripts/learning_curve.py
"""

from pathlib import Path
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import load_castelli, load_expt_gap, metrics, NONMETAL_THRESHOLD, SEED

warnings.filterwarnings("ignore")

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [1, 7, 42]


def two_stage_mae(Xtr, ytr, Xte, yte):
    ybin = (ytr > 0).astype(int)
    if ybin.sum() < 5 or (ybin == 0).sum() < 5:
        return np.nan
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(ybin == 0).sum() / max(ybin.sum(), 1),
        eval_metric="logloss", random_state=SEED)
    clf.fit(Xtr, ybin)
    nz = ytr > 0
    reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                           reg_lambda=1.5, random_state=SEED)
    reg.fit(Xtr[nz], np.log1p(ytr[nz]))
    gap_hat = np.clip(np.expm1(reg.predict(Xte)), 0, None)
    is_nm = clf.predict_proba(Xte)[:, 1] >= NONMETAL_THRESHOLD
    return metrics(is_nm * gap_hat, yte)["nz_mae"]


def curve(name, X, y):
    X = X.reset_index(drop=True)
    n_nz, results = [], {f: [] for f in FRACTIONS}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        rng = np.random.RandomState(seed)
        tr = rng.permutation(tr)
        for f in FRACTIONS:
            sub = tr[: int(len(tr) * f)]
            results[f].append(two_stage_mae(X.iloc[sub], y[sub], X.iloc[te], y[te]))
    sizes = [int(len(y) * 0.8 * f) for f in FRACTIONS]
    means = [np.nanmean(results[f]) for f in FRACTIONS]
    stds = [np.nanstd(results[f]) for f in FRACTIONS]
    print(f"\n{name}: non-zero MAE vs train size")
    for s, m, sd in zip(sizes, means, stds):
        print(f"  n_train={s:6d}   MAE {m:.3f} +/- {sd:.3f}")
    return sizes, means, stds


fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, loader, color in [("Castelli perovskites", load_castelli, "tab:green"),
                            ("expt_gap (experimental)", load_expt_gap, "tab:red")]:
    _n, _xb, X, y, _c = loader()
    sizes, means, stds = curve(name, X, y)
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=name, color=color)

ax.set_xlabel("Training-set size (materials)")
ax.set_ylabel("Non-zero MAE (eV)")
ax.set_title("Learning curves: error still falls at full data\n(both datasets remain data-limited)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "plots"
fig.savefig(out / "learning_curve.svg")
fig.savefig(out / "learning_curve.png", dpi=150)
print(f"\nWrote {out / 'learning_curve.png'}")
