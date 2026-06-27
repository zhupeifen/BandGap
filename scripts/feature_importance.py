"""
SHAP feature importance for the non-zero-gap regressor, per dataset.

Quantifies which descriptors drive the gap prediction (mean |SHAP| over the
non-zero materials), and cross-checks the ablation result (feature_ablation.py)
by aggregating importance into feature groups for Wolverton. Produces a 3-panel
top-feature figure.

    .venv/Scripts/python.exe scripts/feature_importance.py
"""

from pathlib import Path
import warnings

import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import load_wolverton, load_castelli, load_expt_gap, SEED

warnings.filterwarnings("ignore")


def fit_regressor(X, y):
    nz = y > 0
    reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                           reg_lambda=1.5, random_state=SEED)
    reg.fit(X[nz], np.log1p(y[nz]))
    return reg, X[nz]


def shap_importance(reg, Xnz):
    expl = shap.TreeExplainer(reg)
    sv = expl.shap_values(Xnz)
    imp = np.abs(sv).mean(axis=0)
    return imp / imp.sum()                       # normalized mean |SHAP|


def short(name, n=22):
    name = name.replace("MagpieData ", "").replace("lowest distortion_", "dist:")
    return name if len(name) <= n else name[: n - 1] + "…"


datasets = [("Wolverton oxides", load_wolverton),
            ("Castelli perovskites", load_castelli),
            ("expt_gap (experimental)", load_expt_gap)]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, (name, loader) in zip(axes, datasets):
    _n, _xb, X, y, _c = loader()
    reg, Xnz = fit_regressor(X, y)
    imp = shap_importance(reg, Xnz)
    order = np.argsort(imp)[::-1][:8][::-1]
    cols = [short(X.columns[i]) for i in order]
    ax.barh(range(len(order)), imp[order], color="tab:blue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("mean |SHAP| (norm.)", fontsize=9)

fig.suptitle("Top features driving the non-zero band-gap regressor (SHAP)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path(__file__).resolve().parent.parent / "plots"
fig.savefig(out / "feature_importance.svg")
fig.savefig(out / "feature_importance.png", dpi=150)
print(f"Wrote {out / 'feature_importance.png'}")

# --- group-level cross-check for Wolverton -------------------------------
_n, _xb, Xw, yw, _c = load_wolverton()
regw, Xwnz = fit_regressor(Xw, yw)
impw = shap_importance(regw, Xwnz)
energetic = {"e_form", "e_hull", "mu_b", "e_form oxygen"}
structural = {"vpa", "a", "b", "c", "alpha", "beta", "gamma"}
groups = {"energetic (4)": 0.0, "structural+dist": 0.0, "composition/Magpie": 0.0}
for col, w in zip(Xw.columns, impw):
    if col in energetic:
        groups["energetic (4)"] += w
    elif col in structural or str(col).startswith("dist"):
        groups["structural+dist"] += w
    else:
        groups["composition/Magpie"] += w
print("\nWolverton group SHAP importance (share of total):")
for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {g:<22} {v:.1%}")
