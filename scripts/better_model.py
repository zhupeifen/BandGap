"""
Model-improvement experiments (reported honestly, kept only if they help).

The ablation (feature_ablation.py) showed structural columns add little for
Wolverton, but Castelli uses only composition + electronic features even though it
ships pymatgen Structure objects. The main experiment here adds matminer
structure descriptors to Castelli. We also sanity-check probability calibration of
the classifier gate.

    .venv/Scripts/python.exe scripts/better_model.py
"""

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from matminer.utils.io import load_dataframe_from_json
from matminer.featurizers.structure import DensityFeatures

from cv_evaluation import num, magpie, metrics, DATA_DIR, NONMETAL_THRESHOLD, N_SPLITS, SEED

warnings.filterwarnings("ignore")


def two_stage_cv(X, y, calibrate=False):
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    maes, recalls = [], []
    ybin = (y > 0).astype(int)
    for tr, te in kf.split(X):
        spw = (ybin[tr] == 0).sum() / max((ybin[tr] == 1).sum(), 1)
        base = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss", random_state=SEED)
        if calibrate:
            clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
            clf.fit(X.iloc[tr], ybin[tr])
        else:
            base.fit(X.iloc[tr], ybin[tr])
            clf = base
        nz = tr[y[tr] > 0]
        reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                               reg_lambda=1.5, random_state=SEED)
        reg.fit(X.iloc[nz], np.log1p(y[nz]))
        gap_hat = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
        is_nm = clf.predict_proba(X.iloc[te])[:, 1] >= NONMETAL_THRESHOLD
        maes.append(metrics(is_nm * gap_hat, y[te])["nz_mae"])
        recalls.append(np.mean(is_nm[y[te] > 0] == 1))
    return np.array(maes), np.array(recalls)


def density_features(structures):
    df_feat = DensityFeatures()
    rows = []
    for s in structures:
        try:
            rows.append(df_feat.featurize(s))
        except Exception:
            rows.append([np.nan] * len(df_feat.feature_labels()))
    return pd.DataFrame(rows, columns=df_feat.feature_labels())


def main():
    df = load_dataframe_from_json(str(DATA_DIR / "castelli_perovskites.json")).reset_index(drop=True)
    y = num(df["gap gllbsc"]).to_numpy()
    comp = magpie(df["formula"])
    elec = pd.DataFrame({
        "fermi level": num(df["fermi level"]), "fermi width": num(df["fermi width"]),
        "e_form": num(df["e_form"]), "mu_b": num(df["mu_b"]),
        "gap is direct": df["gap is direct"].astype(float)})
    base = pd.concat([comp, elec], axis=1).astype(float)

    print("Featurizing structures (DensityFeatures)...")
    dens = density_features(df["structure"])
    plus_struct = pd.concat([base, dens], axis=1).astype(float)

    print("\n=== Castelli: effect of adding structure (density) features ===")
    for label, X in [("comp + electronic (baseline)", base),
                     ("  + structure (density)", plus_struct)]:
        maes, _ = two_stage_cv(X, y)
        print(f"  {label:<30} non-zero MAE {maes.mean():.3f} +/- {maes.std():.3f}")

    print("\n=== Castelli: classifier calibration (isotonic) at fixed tau ===")
    for label, cal in [("uncalibrated gate", False), ("calibrated gate", True)]:
        maes, rec = two_stage_cv(base, y, calibrate=cal)
        print(f"  {label:<20} non-zero MAE {maes.mean():.3f} +/- {maes.std():.3f}"
              f" | recall {rec.mean():.3f}")


if __name__ == "__main__":
    main()
