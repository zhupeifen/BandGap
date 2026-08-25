"""
Cross-dataset transfer: train the non-zero gap regressor on one dataset and test it
on another, using the common composition-only Magpie feature space. This is the real
deployment test (a model is used on chemistries it was not trained on) and a stronger
generalization probe than any within-dataset split.

Rows = training dataset, columns = test dataset. Diagonal = within-dataset 5-fold
out-of-fold (the optimistic same-distribution number). We report both MAE (eV) and
Pearson r: MAE conflates genuine predictive transfer with the systematic offset between
exchange-correlation functionals (Expt vs PBE vs GLLB-SC), whereas r isolates whether
the model still ranks gaps correctly across datasets.

    .venv/Scripts/python.exe scripts/transfer_matrix.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from matminer.utils.io import load_dataframe_from_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv_evaluation import magpie, num, DATA_DIR, SEED, N_SPLITS

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "plots"
REG = dict(n_estimators=700, max_depth=6, learning_rate=0.03, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.5, random_state=SEED)

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})


def _json_df(fname, gap_col):
    df = load_dataframe_from_json(str(DATA_DIR / fname)).reset_index(drop=True)
    return df["formula"], num(df[gap_col]).to_numpy()


def load_all():
    """Return [(label, formulas, magpie_X (non-zero), y (non-zero))] for each dataset."""
    import pandas as pd
    specs = [
        ("Expt\n(expt.)", *_json_df("expt_gap.json", "gap expt")),
        ("Wolverton\n(PBE)", *_json_df("wolverton_oxides.json", "gap pbe")),
        ("Castelli\n(GLLB-SC)", *_json_df("castelli_perovskites.json", "gap gllbsc")),
    ]
    dp = pd.read_csv(DATA_DIR / "Dataset_double_perovskites_gap_v1.csv")
    specs.append(("Double pero.\n(GLLB-SC)", dp["formula"], num(dp["gap gllbsc"]).to_numpy()))

    out = []
    for label, formulas, y in specs:
        print(f"  featurizing {label.splitlines()[0]} ...")
        X = magpie(formulas).astype(float).reset_index(drop=True)
        y = np.asarray(y, float)
        nz = y > 0
        out.append((label, X[nz].reset_index(drop=True), y[nz]))
    return out


def fit(X, y):
    m = xgb.XGBRegressor(**REG)
    m.fit(X, np.log1p(y))
    return m


def predict(m, X):
    return np.clip(np.expm1(m.predict(X)), 0, None)


# Plot labels (dataset + level of theory), in load_all() order. Also used by the fast
# cache-reload path so figure styling can be tweaked without recomputing the matrices.
LABELS = ["Expt\n(expt.)", "Wolverton\n(PBE)", "Castelli\n(GLLB-SC)", "Dbl. pero.\n(GLLB-SC)"]


def compute():
    data = load_all()
    n = len(data)
    labels = [d[0] for d in data]
    mae = np.zeros((n, n)); rmat = np.zeros((n, n)); mae_bc = np.zeros((n, n))
    offset = np.zeros((n, n))
    for i, (_, Xi, yi) in enumerate(data):
        m_full = fit(Xi, yi)                                   # off-diagonal: train on all of i
        for j, (_, Xj, yj) in enumerate(data):
            if i == j:                                         # within-dataset out-of-fold
                pred = np.zeros_like(yj)
                for tr, te in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xj):
                    pred[te] = predict(fit(Xj.iloc[tr], yj[tr]), Xj.iloc[te])
            else:
                pred = predict(m_full, Xj)
            mae[i, j] = np.mean(np.abs(pred - yj))
            rmat[i, j] = pearsonr(pred, yj)[0] if np.std(pred) > 0 else 0.0
            # Oracle bias correction: remove the mean prediction-minus-truth offset on
            # the test set (the constant shift a functional/label mismatch produces).
            # Uses test labels, so it bounds what any offset calibration could achieve.
            offset[i, j] = np.mean(pred - yj)
            mae_bc[i, j] = np.mean(np.abs(pred - offset[i, j] - yj))
        print(f"  trained on {labels[i].splitlines()[0]:<12} -> "
              + "  ".join(f"{labels[j].splitlines()[0]}:{mae[i,j]:.2f}" for j in range(n)))
    return labels, mae, rmat, mae_bc, offset


def plot(labels, mae, rmat, mae_bc=None):
    n = len(labels)
    npan = 3 if mae_bc is not None else 2
    fig, axes = plt.subplots(1, npan, figsize=(6.4 * npan, 5.4))
    fig.patch.set_alpha(0)
    panels = [("(a) Transfer MAE (eV)", mae, "Reds", "%.2f"),
              ("(b) Transfer Pearson r", rmat, "Greens", "%.2f")]
    if mae_bc is not None:
        panels.append(("(c) Bias-corrected MAE (eV)", mae_bc, "Reds", "%.2f"))
    for ax, (title, M, cmap, fmt) in zip(axes, panels):
        ax.patch.set_alpha(0)
        im = ax.imshow(M, cmap=cmap, aspect="equal",
                       vmin=(0 if "MAE" in title else -0.2),
                       vmax=(np.max(mae) if "MAE" in title else 1.0))  # shared MAE scale
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([l.splitlines()[0] for l in labels], rotation=25, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Tested on", fontsize=20); ax.set_ylabel("Trained on", fontsize=20)
        ax.set_title(title, fontsize=20)
        thr = M.min() + 0.6 * (M.max() - M.min())
        for i in range(n):
            for j in range(n):
                color = "white" if M[i, j] > thr else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                        color=color, fontsize=17, fontweight=weight)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=15)
    fig.tight_layout(w_pad=6.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "transfer_matrix.png", dpi=300, transparent=True, bbox_inches="tight")
    print(f"\nWrote {OUT / 'transfer_matrix.png'}")


def main():
    cache = DATA_DIR / "transfer_matrix_results.json"
    d = json.loads(cache.read_text()) if cache.exists() else {}
    if "mae_bias_corrected" in d:
        labels = LABELS
        mae, rmat = np.array(d["mae"]), np.array(d["pearson_r"])
        mae_bc, offset = np.array(d["mae_bias_corrected"]), np.array(d["mean_offset"])
        print("Loaded cached transfer results "
              "(delete data/transfer_matrix_results.json to recompute).")
    else:
        labels, mae, rmat, mae_bc, offset = compute()
        cache.write_text(json.dumps(
            {"labels": [l.replace("\n", " ") for l in labels],
             "mae": mae.tolist(), "pearson_r": rmat.tolist(),
             "mae_bias_corrected": mae_bc.tolist(), "mean_offset": offset.tolist()},
            indent=2))
    plot(labels, mae, rmat, mae_bc)
    short = [l.replace("\n", " ") for l in labels]
    print("Diagonal (within-dataset OOF) MAE:",
          ", ".join(f"{short[i]} {mae[i, i]:.2f}" for i in range(len(labels))))
    n = len(labels)
    off = ~np.eye(n, dtype=bool)
    print(f"Off-diagonal MAE: raw {mae[off].min():.2f}-{mae[off].max():.2f} eV; "
          f"bias-corrected {mae_bc[off].min():.2f}-{mae_bc[off].max():.2f} eV; "
          f"|mean offset| {np.abs(offset[off]).min():.2f}-{np.abs(offset[off]).max():.2f} eV")
    for i in range(n):
        print("  " + short[i] + " -> " + "  ".join(
            f"{short[j]}: {mae[i, j]:.2f}->{mae_bc[i, j]:.2f} (off {offset[i, j]:+.2f})"
            for j in range(n) if j != i))


if __name__ == "__main__":
    main()
