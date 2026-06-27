"""
Parity plot for the experimental band-gap dataset (expt_gap), two-stage model.

    .venv/Scripts/python.exe scripts/expt_gap_parity.py
"""

from pathlib import Path
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import load_expt_gap, NONMETAL_THRESHOLD, SEED

warnings.filterwarnings("ignore")
TOL = 0.07

name, X, _Xf, y, _cfg = load_expt_gap()
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
ybin = (y > 0).astype(int)

clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=(ybin[tr] == 0).sum() / max((ybin[tr] == 1).sum(), 1),
                        eval_metric="logloss", random_state=SEED)
clf.fit(X.iloc[tr], ybin[tr])
nz = tr[y[tr] > 0]
reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                       reg_lambda=1.5, random_state=SEED)
reg.fit(X.iloc[nz], np.log1p(y[nz]))
gap_hat = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
is_nm = clf.predict_proba(X.iloc[te])[:, 1] >= NONMETAL_THRESHOLD
pred = is_nm * gap_hat

yte = y[te]
mask = yte > 0
true_gap, pred_gap = yte[mask], pred[mask]
gated_off = is_nm[mask] == 0
r2 = 1 - np.sum((true_gap - pred_gap) ** 2) / np.sum((true_gap - true_gap.mean()) ** 2)
mae = np.mean(np.abs(true_gap - pred_gap))

lim = max(true_gap.max(), pred_gap.max()) * 1.05
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal (y = x)")
ax.fill_between([0, lim], [-TOL, lim - TOL], [TOL, lim + TOL],
                color="gray", alpha=0.15, label=f"±{TOL} eV band")
ax.scatter(true_gap[~gated_off], pred_gap[~gated_off], s=14, alpha=0.5,
           color="tab:blue", label="predicted as non-metal")
ax.scatter(true_gap[gated_off], pred_gap[gated_off], s=24, alpha=0.8,
           color="tab:red", marker="x", label="misgated to metal (pred 0)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("True experimental gap [eV]")
ax.set_ylabel("Predicted gap [eV]")
ax.set_title(f"Non-zero band-gap parity (expt_gap)\n"
             f"$R^2$={r2:.3f}   MAE={mae:.3f} eV   n={int(mask.sum())}")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "plots"
fig.savefig(out / "parity_expt_gap.svg")
fig.savefig(out / "parity_expt_gap.png", dpi=150)
print(f"Wrote {out / 'parity_expt_gap.png'}  (R2={r2:.3f} MAE={mae:.3f})")
