"""
Gate analysis for the two-stage band-gap model: how good is the metal/non-metal
classifier, how does the decision threshold tau trade off non-zero accuracy against
misgating, and how much of the residual error is regression error vs. misgating.

Produces (out-of-fold, 5-fold CV, no leakage) for the three zero-inflated datasets:
  (a) reliability diagram (calibration of P(non-metal))
  (b) ROC curve + AUC
  (c) tau sweep: non-zero MAE vs. the gate threshold (marking tau = 0.25)
  (d) error decomposition at tau = 0.25: regression error vs. misgating penalty

    .venv/Scripts/python.exe scripts/gate_analysis.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import load_wolverton, load_castelli, load_expt_gap, SEED, N_SPLITS

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "plots"
TAU = 0.25
TAU_GRID = np.linspace(0.02, 0.90, 45)
COLORS = {"Wolverton oxides": "#1f77b4", "Castelli perovskites": "#d62728",
          "Expt gap (experimental)": "#2ca02c"}
SHORT = {"Wolverton oxides": "Wolverton", "Castelli perovskites": "Castelli",
         "Expt gap (experimental)": "Expt gap"}

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 20,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})


def out_of_fold(X_full, y):
    """Return OOF P(non-metal) and OOF predicted gap for every material."""
    ybin = (y > 0).astype(int)
    p = np.full(len(y), np.nan)
    gap_hat = np.full(len(y), np.nan)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X_full):
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(ybin[tr] == 0).sum() / max((ybin[tr] == 1).sum(), 1),
            eval_metric="logloss", random_state=SEED)
        clf.fit(X_full.iloc[tr], ybin[tr])
        p[te] = clf.predict_proba(X_full.iloc[te])[:, 1]

        nz_tr = tr[y[tr] > 0]
        reg = xgb.XGBRegressor(
            n_estimators=700, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            reg_lambda=1.5, random_state=SEED)
        reg.fit(X_full.iloc[nz_tr], np.log1p(y[nz_tr]))
        gap_hat[te] = np.clip(np.expm1(reg.predict(X_full.iloc[te])), 0, None)
    return p, gap_hat, ybin


def tau_curve(p, gap_hat, y):
    """Non-zero MAE and misgating rate as a function of the gate threshold."""
    nz = y > 0
    maes, misgate = [], []
    for t in TAU_GRID:
        is_nm = p >= t
        pred = is_nm * gap_hat
        maes.append(np.mean(np.abs(pred - y)[nz]))
        misgate.append(np.mean((~is_nm)[nz]))      # true non-zero gated to metal
    return np.array(maes), np.array(misgate)


def decompose(p, gap_hat, y, tau=TAU):
    """Split non-zero MAE into regression error (correctly gated) and misgating penalty."""
    nz = y > 0
    is_nm = p >= tau
    n_nz = nz.sum()
    correct = nz & is_nm
    misgated = nz & ~is_nm
    reg_component = np.sum(np.abs(gap_hat - y)[correct]) / n_nz
    misgate_component = np.sum(y[misgated]) / n_nz       # predicted 0 -> error = true gap
    return reg_component, misgate_component


def main():
    results = {}
    datasets = []
    for loader in (load_wolverton, load_castelli, load_expt_gap):
        name, _Xb, X_full, y, _cfg = loader()
        print(f"  {name}: out-of-fold classify+regress ...")
        p, gap_hat, ybin = out_of_fold(X_full, y)
        fpr, tpr, _ = roc_curve(ybin, p)
        roc_auc = auc(fpr, tpr)
        frac_pos, mean_pred = calibration_curve(ybin, p, n_bins=10, strategy="quantile")
        maes, misgate = tau_curve(p, gap_hat, y)
        reg_c, mis_c = decompose(p, gap_hat, y, TAU)
        datasets.append(name)
        results[name] = dict(
            auc=float(roc_auc), reg_component=float(reg_c), misgate_component=float(mis_c),
            nz_mae_at_tau=float(reg_c + mis_c),
            misgate_rate_at_tau=float(np.mean((p[y > 0] < TAU))),
            fpr=fpr.tolist(), tpr=tpr.tolist(),
            frac_pos=frac_pos.tolist(), mean_pred=mean_pred.tolist(),
            tau_grid=TAU_GRID.tolist(), tau_mae=maes.tolist(), tau_misgate=misgate.tolist())

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0))
    fig.patch.set_alpha(0)
    for ax in axes.ravel():
        ax.patch.set_alpha(0)
        for s in ax.spines.values():
            s.set_linewidth(1.0)

    # (a) reliability
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in datasets:
        r = results[name]
        ax.plot(r["mean_pred"], r["frac_pos"], "o-", color=COLORS[name],
                ms=4, lw=1.4, label=SHORT[name])
    ax.set_xlabel("Mean predicted P(non-metal)"); ax.set_ylabel("Observed non-metal fraction")
    ax.set_title("(a) Classifier calibration")
    ax.legend(frameon=False)

    # (b) ROC
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name in datasets:
        r = results[name]
        ax.plot(r["fpr"], r["tpr"], color=COLORS[name], lw=1.6,
                label=f"{SHORT[name]} (AUC {r['auc']:.3f})")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.set_title("(b) ROC of the metal/non-metal gate")
    ax.legend(frameon=False, loc="lower right")

    # (c) tau sweep (non-zero MAE)
    ax = axes[1, 0]
    for name in datasets:
        r = results[name]
        ax.plot(r["tau_grid"], r["tau_mae"], color=COLORS[name], lw=1.6, label=SHORT[name])
    ax.axvline(TAU, color="0.4", ls=":", lw=1.2)
    ax.text(TAU + 0.01, ax.get_ylim()[1], r" $\tau=0.25$", va="top", fontsize=15, color="0.3")
    ax.set_xlabel(r"Gate threshold $\tau$"); ax.set_ylabel("Non-zero MAE (eV)")
    ax.set_title("(c) Threshold sensitivity")
    ax.legend(frameon=False)

    # (d) error decomposition at tau
    ax = axes[1, 1]
    labels = [SHORT[n] for n in datasets]
    reg_c = [results[n]["reg_component"] for n in datasets]
    mis_c = [results[n]["misgate_component"] for n in datasets]
    x = np.arange(len(datasets))
    ax.bar(x, reg_c, 0.55, label="regression error", color="#4c72b0")
    ax.bar(x, mis_c, 0.55, bottom=reg_c, label="misgating penalty", color="#dd8452")
    for xi, (rc, mc) in enumerate(zip(reg_c, mis_c)):
        ax.text(xi, rc + mc, f"{rc+mc:.3f}", ha="center", va="bottom", fontsize=15)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Non-zero MAE (eV)")
    ax.set_title(r"(d) Error decomposition at $\tau=0.25$")
    ax.legend(frameon=False)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "gate_analysis.png", dpi=300, transparent=True, bbox_inches="tight")
    (OUT.parent / "data" / "gate_analysis_results.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items()
                        if kk in ("auc", "reg_component", "misgate_component",
                                  "nz_mae_at_tau", "misgate_rate_at_tau")}
                    for k, v in results.items()}, indent=2))
    print(f"\nWrote {OUT / 'gate_analysis.png'}")
    for name in datasets:
        r = results[name]
        print(f"  {SHORT[name]:<10} AUC={r['auc']:.3f}  nzMAE={r['nz_mae_at_tau']:.3f} "
              f"(reg {r['reg_component']:.3f} + misgate {r['misgate_component']:.3f})  "
              f"misgate-rate={r['misgate_rate_at_tau']:.3f}")


if __name__ == "__main__":
    main()
