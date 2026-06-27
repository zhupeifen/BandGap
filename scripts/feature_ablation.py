"""
Feature-group ablation + classifier error analysis.

Motivation: the nested-CV result (nested_cv.py) shows that with features held
constant, the two-stage *method* contributes only a modest MAE reduction. This
script decomposes where the predictive signal actually lives, by running the
two-stage model in 5-fold CV over feature-group subsets:

  Wolverton oxides : composition (Magpie) / energetic / structural / combinations
  Castelli         : composition (Magpie) / electronic / combination

It also reports the gap distribution of the materials the classifier *misgates*
(true non-zero predicted as metal), to characterize where gating fails.

    .venv/Scripts/python.exe scripts/feature_ablation.py
"""

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from matminer.utils.io import load_dataframe_from_json

from cv_evaluation import num, magpie, metrics, DATA_DIR, NONMETAL_THRESHOLD, N_SPLITS, SEED

warnings.filterwarnings("ignore")


def two_stage_cv(X, y, collect_misgated=False):
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    maes, misgated_gaps = [], []
    ybin = (y > 0).astype(int)
    for tr, te in kf.split(X):
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
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
        maes.append(metrics(is_nm * gap_hat, y[te])["nz_mae"])
        if collect_misgated:
            te_nz = y[te] > 0
            misgated = te_nz & (~is_nm)
            misgated_gaps.extend(y[te][misgated].tolist())
    return np.array(maes), misgated_gaps


def ablate(name, groups, y, combos):
    print(f"\n=== {name}: feature-group ablation (two-stage, {N_SPLITS}-fold, non-zero MAE) ===")
    print(f"{'feature set':<28}{'n_feat':>8}{'non-zero MAE':>18}")
    for label, keys in combos:
        X = pd.concat([groups[k] for k in keys], axis=1).astype(float)
        maes, _ = two_stage_cv(X, y)
        print(f"{label:<28}{X.shape[1]:>8}{maes.mean():>10.3f} +/-{maes.std():<6.3f}")


def run_wolverton():
    df = load_dataframe_from_json(str(DATA_DIR / "wolverton_oxides.json")).reset_index(drop=True)
    y = num(df["gap pbe"]).to_numpy()
    groups = {
        "comp": magpie(df["formula"]),
        "energetic": pd.DataFrame({c: num(df[c]) for c in
                                   ["e_form", "e_hull", "mu_b", "e_form oxygen"]}),
        "struct": pd.concat([
            pd.DataFrame({c: num(df[c]) for c in ["a", "b", "c", "alpha", "beta", "gamma", "vpa"]}),
            pd.get_dummies(df["lowest distortion"].astype(str), prefix="dist")], axis=1),
    }
    combos = [("composition only", ["comp"]),
              ("energetic only", ["energetic"]),
              ("structural only", ["struct"]),
              ("composition+energetic", ["comp", "energetic"]),
              ("composition+structural", ["comp", "struct"]),
              ("all (comp+energ+struct)", ["comp", "energetic", "struct"])]
    ablate("Wolverton oxides", groups, y, combos)
    # error analysis on the full feature set
    Xall = pd.concat([groups[k] for k in ("comp", "energetic", "struct")], axis=1).astype(float)
    _, misgated = two_stage_cv(Xall, y, collect_misgated=True)
    nz = y[y > 0]
    print(f"  misgated (true non-zero -> metal): {len(misgated)} materials; "
          f"median gap {np.median(misgated):.2f} eV vs {np.median(nz):.2f} eV overall; "
          f"{100*np.mean(np.array(misgated) < 1.0):.0f}% are <1 eV "
          f"(vs {100*np.mean(nz < 1.0):.0f}% of all non-zero)")


def run_castelli():
    df = load_dataframe_from_json(str(DATA_DIR / "castelli_perovskites.json")).reset_index(drop=True)
    y = num(df["gap gllbsc"]).to_numpy()
    groups = {
        "comp": magpie(df["formula"]),
        "electronic": pd.DataFrame({
            "fermi level": num(df["fermi level"]), "fermi width": num(df["fermi width"]),
            "e_form": num(df["e_form"]), "mu_b": num(df["mu_b"]),
            "gap is direct": df["gap is direct"].astype(float)}),
    }
    combos = [("composition only", ["comp"]),
              ("electronic only", ["electronic"]),
              ("all (comp+electronic)", ["comp", "electronic"])]
    ablate("Castelli perovskites", groups, y, combos)
    Xall = pd.concat([groups[k] for k in ("comp", "electronic")], axis=1).astype(float)
    _, misgated = two_stage_cv(Xall, y, collect_misgated=True)
    nz = y[y > 0]
    print(f"  misgated (true non-zero -> metal): {len(misgated)} materials; "
          f"median gap {np.median(misgated):.2f} eV vs {np.median(nz):.2f} eV overall; "
          f"{100*np.mean(np.array(misgated) < 1.0):.0f}% are <1 eV "
          f"(vs {100*np.mean(nz < 1.0):.0f}% of all non-zero)")


if __name__ == "__main__":
    run_wolverton()
    run_castelli()
