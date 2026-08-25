"""
Calibrated prediction intervals for the non-zero band-gap regressor via split
conformal prediction. A screening user needs not just a point gap but a trustworthy
uncertainty; split conformal gives finite-sample marginal coverage with no
distributional assumptions.

For each dataset (non-zero materials, enriched features) we run 5-fold CV; within each
fold the training portion is split 75/25 into a fit set and a calibration set. The
calibration residuals |y - y_hat| (eV) set the interval half-width q at each nominal
level, and we measure the empirical coverage and interval width on the test fold.

    .venv/Scripts/python.exe scripts/conformal_uncertainty.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import load_wolverton, load_castelli, load_expt_gap, SEED, N_SPLITS

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "plots"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEVELS = np.round(np.arange(0.50, 0.96, 0.05), 2)
COLORS = {"Wolverton oxides": "#1f77b4", "Castelli perovskites": "#d62728",
          "Expt gap (experimental)": "#2ca02c"}
SHORT = {"Wolverton oxides": "Wolverton", "Castelli perovskites": "Castelli",
         "Expt gap (experimental)": "Expt gap"}

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 20,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})


def reg():
    return xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                            reg_lambda=1.5, random_state=SEED)


def conformal(X, y):
    """Return {level: (empirical_coverage, mean_width)} and the 90% half-width."""
    nz = y > 0
    Xn = X[nz].reset_index(drop=True); yn = y[nz]
    cov = {L: [] for L in LEVELS}
    wid = {L: [] for L in LEVELS}
    q90 = []
    for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xn):
        fit_i, cal_i = train_test_split(tr, test_size=0.25, random_state=SEED)
        m = reg(); m.fit(Xn.iloc[fit_i], np.log1p(yn[fit_i]))
        pred_cal = np.clip(np.expm1(m.predict(Xn.iloc[cal_i])), 0, None)
        scores = np.abs(yn[cal_i] - pred_cal)
        pred_te = np.clip(np.expm1(m.predict(Xn.iloc[te])), 0, None)
        n = len(scores)
        for L in LEVELS:
            k = min(int(np.ceil((n + 1) * L)), n)
            q = np.sort(scores)[k - 1]
            lo = np.clip(pred_te - q, 0, None); hi = pred_te + q
            cov[L].append(np.mean((yn[te] >= lo) & (yn[te] <= hi)))
            wid[L].append(np.mean(hi - lo))
            if abs(L - 0.90) < 1e-9:
                q90.append(q)
    emp = {float(L): (float(np.mean(cov[L])), float(np.mean(wid[L]))) for L in LEVELS}
    return emp, float(np.mean(q90))


def main():
    results, names = {}, []
    for loader in (load_wolverton, load_castelli, load_expt_gap):
        name, _xb, X_full, y, _c = loader()
        print(f"  {name}: split-conformal 5-fold ...")
        emp, q90 = conformal(X_full.astype(float), y)
        results[name] = {"coverage": emp, "half_width_90": q90}
        names.append(name)
        c90 = emp[0.9][0]
        print(f"    90% interval: empirical coverage {c90:.3f}, half-width +/-{q90:.3f} eV")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax in axes:
        ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)

    ax = axes[0]
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="ideal")
    for name in names:
        L = sorted(results[name]["coverage"])
        emp = [results[name]["coverage"][l][0] for l in L]
        ax.plot(L, emp, "o-", color=COLORS[name], ms=4, lw=1.5, label=SHORT[name])
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) Conformal calibration")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    labels = [SHORT[n] for n in names]
    hw = [results[n]["half_width_90"] for n in names]
    x = np.arange(len(names))
    ax.bar(x, hw, 0.55, color=[COLORS[n] for n in names])
    for xi, h in zip(x, hw):
        ax.text(xi, h, f"±{h:.2f}", ha="center", va="bottom", fontsize=15)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("90% interval half-width (eV)")
    ax.set_title("(b) Interval sharpness (90%)")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "conformal_uncertainty.png", dpi=300, transparent=True, bbox_inches="tight")
    (DATA_DIR / "conformal_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT / 'conformal_uncertainty.png'}")


if __name__ == "__main__":
    main()
