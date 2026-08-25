"""
CrabNet (composition transformer, Wang et al. 2021) vs. the gradient-boosted trees,
on the SAME three non-zero CV datasets, same 5-fold splits. CrabNet learns its own
representation directly from the chemical formula, so this tests whether a dedicated
composition *deep* model breaks the gradient-boosting accuracy ceiling. Reports
non-zero MAE (eV), mean +/- std over folds.

    python crabnet_compare.py --csvdir crabnet_csv --epochs 300
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 42
N_SPLITS = 5
SETS = ["wolverton", "castelli", "expt_gap"]


def get_crabnet(epochs):
    from crabnet.crabnet_ import CrabNet
    return CrabNet(epochs=epochs, verbose=False)


def predict_array(cb, test_df):
    out = cb.predict(test_df)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return np.asarray(out, dtype=float).reshape(-1)


def mae_cv(csv, epochs):
    df = pd.read_csv(csv)[["formula", "target"]].dropna().reset_index(drop=True)
    maes = []
    for k, (tr, te) in enumerate(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(df)):
        train = df.iloc[tr].reset_index(drop=True)
        test = df.iloc[te].reset_index(drop=True)
        cb = get_crabnet(epochs)
        cb.fit(train)
        pred = np.clip(predict_array(cb, test), 0, None)
        diff = np.abs(pred - test["target"].to_numpy())
        finite = np.isfinite(diff)              # CrabNet returns nan for formulas it cannot parse
        nbad = int((~finite).sum())
        mae = float(np.mean(diff[finite]))
        maes.append(mae)
        tag = f" ({nbad} unparseable dropped)" if nbad else ""
        print(f"    {os.path.basename(csv)} fold {k}: MAE {mae:.3f} eV{tag}", flush=True)
    return float(np.mean(maes)), float(np.std(maes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvdir", default="crabnet_csv")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--only", default="", help="comma-separated dataset names to run (default: all)")
    args = ap.parse_args()
    sets = [s for s in SETS if not args.only or s in args.only.split(",")]

    try:
        import torch
        print(f"torch {torch.__version__}; cuda={torch.cuda.is_available()}")
    except Exception as e:
        print("torch import:", e)

    print(f"CrabNet non-zero MAE (eV), {N_SPLITS}-fold CV, {args.epochs} epochs\n")
    for name in sets:
        csv = os.path.join(args.csvdir, f"{name}.csv")
        if not os.path.exists(csv):
            print(f"{name}: MISSING {csv}"); continue
        mu, sd = mae_cv(csv, args.epochs)
        print(f"  {name:<12} CrabNet: {mu:.3f} +/- {sd:.3f}\n", flush=True)


if __name__ == "__main__":
    main()
