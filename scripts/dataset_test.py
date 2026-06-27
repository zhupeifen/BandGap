"""
Band-gap prediction on the Castelli perovskites dataset (target: gap gllbsc).

This dataset is extremely zero-inflated: 18,193 of 18,928 materials (96.1%) have
gap = 0, and only 735 have a non-zero gap. The original single-stage regressor
trained on all of them, so it learned to predict ~0 and scored a deceptively high
0.92 accuracy (mostly from correctly calling metals 0) while doing poorly on the
materials that actually have a gap. That original model is kept below as BASELINE.

The improved model applies the same fix used on the Wolverton scripts:
  1. A metal / non-metal classifier gates the prediction (scale_pos_weight handles
     the 4%/96% imbalance; threshold lowered to recover misgated small-gap cases).
  2. The regressor is trained ONLY on the 735 non-zero materials, on a log1p target.
  3. Richer features: Magpie elemental descriptors from the formula + the original
     numeric features (fermi level/width, e_form, mu_b, gap is direct).

NOTE: 'vbm' and 'cbm' are deliberately excluded as features -- gap = cbm - vbm,
so using them would be target leakage (the original script also avoided them).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split

from pymatgen.core import Composition
from matminer.utils.io import load_dataframe_from_json
from matminer.featurizers.composition import ElementProperty

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM = 53                 # original script's seed
TOL = 0.07
NONMETAL_THRESHOLD = 0.25   # recover misgated small-gap materials

df = load_dataframe_from_json(str(DATA_DIR / "castelli_perovskites.json")).reset_index(drop=True)


def num(series):
    return pd.to_numeric(series, errors="coerce")


y = num(df["gap gllbsc"]).to_numpy()
y_bin = (y > 0).astype(int)
print(f"materials: {len(df)}   non-zero gaps: {int(y_bin.sum())} "
      f"({y_bin.mean():.1%})")

# Original numeric features (NO vbm/cbm -> they define the gap = leakage)
base_num = pd.DataFrame({
    "fermi level": num(df["fermi level"]),
    "fermi width": num(df["fermi width"]),
    "e_form": num(df["e_form"]),
    "mu_b": num(df["mu_b"]),
    "gap is direct": df["gap is direct"].astype(float),
}).astype(float)

# BASELINE feature matrix = the original 5 features
X_base = base_num

# IMPROVED feature matrix = original 5 + Magpie composition descriptors
print("Featurizing compositions (Magpie)...")
ep = ElementProperty.from_preset("magpie")
comp_rows = []
for f in df["formula"]:
    try:
        comp_rows.append(ep.featurize(Composition(str(f))))
    except Exception:
        comp_rows.append([np.nan] * len(ep.feature_labels()))
comp = pd.DataFrame(comp_rows, columns=ep.feature_labels())
X_full = pd.concat([base_num, comp], axis=1).astype(float)

# Shared split
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM)
y_tr, y_te = y[tr], y[te]
nz_test_mask = y_te > 0


def report(name, pred):
    err = np.abs(pred - y_te)
    nz = nz_test_mask
    print(f"\n=== {name} ===")
    print(f"  overall acc (<{TOL} eV)   : {np.mean(err < TOL):.3f}")
    print(f"  NON-ZERO acc (<{TOL} eV)  : {np.mean(err[nz] < TOL):.3f}   <-- target")
    print(f"  NON-ZERO acc (<0.30 eV)   : {np.mean(err[nz] < 0.30):.3f}")
    print(f"  NON-ZERO MAE  (eV)        : {np.mean(err[nz]):.3f}")
    print(f"  NON-ZERO RMSE (eV)        : {np.sqrt(np.mean(err[nz] ** 2)):.3f}")


# (A) BASELINE: original single-stage regressor on ALL data
baseline = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=RANDOM)
baseline.fit(X_base.iloc[tr], y_tr)
pred_base = baseline.predict(X_base.iloc[te])
report("BASELINE (original: single regressor, all data, 5 features)", pred_base)

# (B) IMPROVED: classifier gate + non-zero log regressor + Magpie features
clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_bin[tr] == 0).sum() / max((y_bin[tr] == 1).sum(), 1),
    objective="binary:logistic", eval_metric="logloss", random_state=RANDOM)
clf.fit(X_full.iloc[tr], y_bin[tr])

nz_tr = tr[y[tr] > 0]
reg = xgb.XGBRegressor(
    n_estimators=700, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
    reg_lambda=1.5, objective="reg:squarederror", random_state=RANDOM)
reg.fit(X_full.loc[nz_tr], np.log1p(y[nz_tr]))

gap_hat = np.clip(np.expm1(reg.predict(X_full.iloc[te])), 0, None)
proba = clf.predict_proba(X_full.iloc[te])[:, 1]
is_nonmetal = proba >= NONMETAL_THRESHOLD
pred_imp = is_nonmetal * gap_hat
report(f"IMPROVED (two-stage, non-zero reg, thr={NONMETAL_THRESHOLD})", pred_imp)
print(f"\nClassifier recall on non-zero test materials: "
      f"{np.mean(is_nonmetal[nz_test_mask] == 1):.3f}")

# ---------------------------------------------------------------------------
# Plot 1 (original): prediction error vs. index for the improved model
# ---------------------------------------------------------------------------
hit = (np.abs(pred_imp - y_te) < TOL).astype(int)
data = pd.DataFrame({
    "index": np.arange(len(y_te)),
    "y_test": y_te,
    "difference": y_te - pred_imp,
    "accuracy": hit,
})
sns.set_theme()
sns.relplot(data=data, x="index", y="difference", hue="y_test", style="accuracy")

# ---------------------------------------------------------------------------
# Plot 2 (new): parity plot for the true non-zero materials, improved model
# ---------------------------------------------------------------------------
true_gap = y_te[nz_test_mask]
pred_gap = pred_imp[nz_test_mask]
gated_off = is_nonmetal[nz_test_mask] == 0
ss_res = np.sum((true_gap - pred_gap) ** 2)
ss_tot = np.sum((true_gap - true_gap.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
mae = np.mean(np.abs(true_gap - pred_gap))

lim = max(true_gap.max(), pred_gap.max()) * 1.05
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal (y = x)")
ax.fill_between([0, lim], [-TOL, lim - TOL], [TOL, lim + TOL],
                color="gray", alpha=0.15, label=f"±{TOL} eV band")
ax.scatter(true_gap[~gated_off], pred_gap[~gated_off], s=18, alpha=0.6,
           color="tab:blue", label="predicted as non-metal")
ax.scatter(true_gap[gated_off], pred_gap[gated_off], s=28, alpha=0.8,
           color="tab:red", marker="x", label="misgated to metal (pred 0)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("True gap (gllbsc) [eV]")
ax.set_ylabel("Predicted gap [eV]")
ax.set_title(f"Non-zero band-gap parity (dataset_test / Castelli)\n"
             f"$R^2$={r2:.3f}   MAE={mae:.3f} eV   n={int(nz_test_mask.sum())}")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()

plots_dir = Path(__file__).resolve().parent.parent / "plots"
fig.savefig(plots_dir / "parity_dataset_test.svg")
fig.savefig(plots_dir / "parity_dataset_test.png", dpi=150)
print(f"\nParity plot saved to {plots_dir / 'parity_dataset_test.png'}")

plt.show()
