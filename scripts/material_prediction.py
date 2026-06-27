"""
Band-gap prediction on the Wolverton oxides dataset.

The original version of this script trained a SINGLE regressor on all materials
(79% of which have gap = 0), so it was pulled toward predicting ~0 and did poorly
on the materials that actually have a gap. That original model is kept below as
the BASELINE so the improvement is visible head-to-head.

The improved model applies the same fix used in binary_band_gap_improved.py
(NOTE: same Wolverton dataset as that script):
  1. A metal / non-metal classifier gates the prediction.
  2. The regressor is trained ONLY on non-zero-gap materials, on a log1p target.
  3. Richer features: Magpie elemental descriptors from the formula + structural
     columns (lattice params + angles, vpa, e_form, e_hull, mu_b, e_form oxygen,
     one-hot 'lowest distortion').
  4. The classifier threshold is lowered to 0.25 to recover misgated small-gap
     materials (a true non-zero material gated to 0 is a guaranteed miss).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split

from pymatgen.core import Element, Composition
from matminer.utils.io import load_dataframe_from_json
from matminer.featurizers.composition import ElementProperty

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM = 46                 # original script's seed
TOL = 0.07                  # eV, the original accuracy threshold
NONMETAL_THRESHOLD = 0.25   # recover misgated small-gap materials

df = load_dataframe_from_json(str(DATA_DIR / "wolverton_oxides.json")).reset_index(drop=True)


def num(series):
    """Coerce a column to float, mapping '-' and bad values to NaN."""
    return pd.to_numeric(series, errors="coerce")


y = num(df["gap pbe"]).to_numpy()
y_bin = (y > 0).astype(int)
print(f"materials: {len(df)}   non-zero gaps: {int(y_bin.sum())} "
      f"({y_bin.mean():.1%})")

# ---------------------------------------------------------------------------
# BASELINE feature matrix: the original 9 hand-built features
# ---------------------------------------------------------------------------
def safe_Z(sym):
    try:
        return Element(str(sym)).Z
    except Exception:
        return np.nan

X_base = pd.DataFrame({
    "Z_a": df["atom a"].map(safe_Z),
    "Z_b": df["atom b"].map(safe_Z),
    "mu_b": num(df["mu_b"]).fillna(0),
    "e_form": num(df["e_form"]),
    "a": num(df["a"]), "b": num(df["b"]), "c": num(df["c"]),
    "e_hull": num(df["e_hull"]),
    "e_form_oxygen": num(df["e_form oxygen"]),
}).astype(float)

# ---------------------------------------------------------------------------
# IMPROVED feature matrix: structural columns + Magpie composition descriptors
# ---------------------------------------------------------------------------
struct_cols = ["e_form", "e_hull", "mu_b", "vpa", "a", "b", "c",
               "alpha", "beta", "gamma", "e_form oxygen"]
struct = pd.DataFrame({c: num(df[c]) for c in struct_cols})
distortion = pd.get_dummies(df["lowest distortion"].astype(str), prefix="dist")

print("Featurizing compositions (Magpie)...")
ep = ElementProperty.from_preset("magpie")
comp_rows = []
for f in df["formula"]:
    try:
        comp_rows.append(ep.featurize(Composition(str(f))))
    except Exception:
        comp_rows.append([np.nan] * len(ep.feature_labels()))
comp = pd.DataFrame(comp_rows, columns=ep.feature_labels())
X_full = pd.concat([struct, distortion, comp], axis=1).astype(float)

# ---------------------------------------------------------------------------
# Shared split + reporting
# ---------------------------------------------------------------------------
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
    n_estimators=300, max_depth=9, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=RANDOM)
baseline.fit(X_base.iloc[tr], y_tr)
pred_base = baseline.predict(X_base.iloc[te])
report("BASELINE (original: single regressor, all data, 9 features)", pred_base)

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
    "test data": y_te,
    "difference (test data - ai prediction)": y_te - pred_imp,
    "accuracy": hit,
})
sns.set_theme()
sns.relplot(data=data, x="index", y="difference (test data - ai prediction)",
            hue="test data", style="accuracy")

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
ax.set_xlabel("True gap (PBE) [eV]")
ax.set_ylabel("Predicted gap [eV]")
ax.set_title(f"Non-zero band-gap parity (material_prediction)\n"
             f"$R^2$={r2:.3f}   MAE={mae:.3f} eV   n={int(nz_test_mask.sum())}")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()

plots_dir = Path(__file__).resolve().parent.parent / "plots"
fig.savefig(plots_dir / "parity_material_prediction.svg")
fig.savefig(plots_dir / "parity_material_prediction.png", dpi=150)
print(f"\nParity plot saved to {plots_dir / 'parity_material_prediction.png'}")

plt.show()
