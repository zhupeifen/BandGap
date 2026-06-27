"""
Composition-only Magpie + Random-Forest baseline (5-fold CV).

Context for the manuscript: a common quick baseline in materials informatics is a
Random Forest on Magpie composition descriptors alone (no structural/energetic
features). This script evaluates that baseline in the SAME 5-fold CV harness as
cv_evaluation.py, in both single-stage and two-stage form, so the numbers are
directly comparable to the XGBoost two-stage model (Table 2 of the manuscript).

    .venv/Scripts/python.exe scripts/cv_rf_baseline.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from matminer.utils.io import load_dataframe_from_json

# Reuse helpers/constants from the XGBoost CV script (same dir on sys.path).
from cv_evaluation import num, magpie, metrics, DATA_DIR, TOL, NONMETAL_THRESHOLD, N_SPLITS, SEED

warnings.filterwarnings("ignore")


def load_magpie_only(filename, target):
    df = load_dataframe_from_json(str(DATA_DIR / filename)).reset_index(drop=True)
    y = num(df[target]).to_numpy()
    X = magpie(df["formula"]).astype(float)          # composition descriptors only
    return X, y


def rf_reg():
    return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)


def run(name, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    single, two = [], []
    ybin = (y > 0).astype(int)
    for tr, te in kf.split(X):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        Xtr, Xte = imp.transform(X.iloc[tr]), imp.transform(X.iloc[te])
        y_tr, y_te = y[tr], y[te]

        # Single-stage RF on all data (the naive composition-only baseline)
        m = rf_reg().fit(Xtr, y_tr)
        single.append(metrics(m.predict(Xte), y_te))

        # Two-stage RF: balanced classifier gate + non-zero log regressor
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, ybin[tr])
        nz = y_tr > 0
        reg = rf_reg().fit(Xtr[nz], np.log1p(y_tr[nz]))
        gap_hat = np.clip(np.expm1(reg.predict(Xte)), 0, None)
        is_nm = clf.predict_proba(Xte)[:, 1] >= NONMETAL_THRESHOLD
        two.append(metrics(is_nm * gap_hat, y_te))

    def agg(rows, key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std()

    print(f"\n=== {name}  (Magpie-only RF, {N_SPLITS}-fold CV, mean +/- std) ===")
    print(f"{'metric':<22}{'RF single-stage':>20}{'RF two-stage':>20}")
    for key, label in [("nz_mae", "non-zero MAE (eV)"),
                       ("nz_rmse", "non-zero RMSE (eV)"),
                       ("nz_acc_007", "non-zero acc<0.07"),
                       ("nz_acc_030", "non-zero acc<0.30"),
                       ("overall_acc", "overall acc<0.07")]:
        sm, ss = agg(single, key)
        tm, ts = agg(two, key)
        print(f"{label:<22}{sm:>10.3f} +/-{ss:<6.3f}{tm:>10.3f} +/-{ts:<6.3f}")


if __name__ == "__main__":
    run("Wolverton oxides", *load_magpie_only("wolverton_oxides.json", "gap pbe"))
    run("Castelli perovskites", *load_magpie_only("castelli_perovskites.json", "gap gllbsc"))
