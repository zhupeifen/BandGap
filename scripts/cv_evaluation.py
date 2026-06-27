"""
5-fold cross-validated evaluation: single-stage baseline vs. two-stage model.

Reports mean +/- std across folds for the non-zero-material metrics, for the two
zero-inflated datasets (Wolverton oxides, Castelli perovskites). Produces the
error-barred numbers needed for the manuscript (replaces single-split results).

    .venv/Scripts/python.exe scripts/cv_evaluation.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from pymatgen.core import Element, Composition
from matminer.utils.io import load_dataframe_from_json
from matminer.featurizers.composition import ElementProperty

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOL = 0.07
NONMETAL_THRESHOLD = 0.25
N_SPLITS = 5
SEED = 42

_EP = ElementProperty.from_preset("magpie")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def magpie(formulas):
    rows = []
    for f in formulas:
        try:
            rows.append(_EP.featurize(Composition(str(f))))
        except Exception:
            rows.append([np.nan] * len(_EP.feature_labels()))
    return pd.DataFrame(rows, columns=_EP.feature_labels())


def safe_Z(sym):
    try:
        return Element(str(sym)).Z
    except Exception:
        return np.nan


def load_wolverton():
    df = load_dataframe_from_json(str(DATA_DIR / "wolverton_oxides.json")).reset_index(drop=True)
    y = num(df["gap pbe"]).to_numpy()
    X_base = pd.DataFrame({
        "Z_a": df["atom a"].map(safe_Z), "Z_b": df["atom b"].map(safe_Z),
        "mu_b": num(df["mu_b"]).fillna(0), "e_form": num(df["e_form"]),
        "a": num(df["a"]), "b": num(df["b"]), "c": num(df["c"]),
        "e_hull": num(df["e_hull"]), "e_form_oxygen": num(df["e_form oxygen"]),
    }).astype(float)
    struct_cols = ["e_form", "e_hull", "mu_b", "vpa", "a", "b", "c",
                   "alpha", "beta", "gamma", "e_form oxygen"]
    struct = pd.DataFrame({c: num(df[c]) for c in struct_cols})
    dist = pd.get_dummies(df["lowest distortion"].astype(str), prefix="dist")
    X_full = pd.concat([struct, dist, magpie(df["formula"])], axis=1).astype(float)
    base_cfg = dict(n_estimators=300, max_depth=9, learning_rate=0.05)
    return "Wolverton oxides", X_base, X_full, y, base_cfg


def load_castelli():
    df = load_dataframe_from_json(str(DATA_DIR / "castelli_perovskites.json")).reset_index(drop=True)
    y = num(df["gap gllbsc"]).to_numpy()
    base_num = pd.DataFrame({
        "fermi level": num(df["fermi level"]), "fermi width": num(df["fermi width"]),
        "e_form": num(df["e_form"]), "mu_b": num(df["mu_b"]),
        "gap is direct": df["gap is direct"].astype(float),
    }).astype(float)
    X_full = pd.concat([base_num, magpie(df["formula"])], axis=1).astype(float)
    base_cfg = dict(n_estimators=300, max_depth=6, learning_rate=0.05)
    return "Castelli perovskites", base_num, X_full, y, base_cfg


def load_expt_gap():
    # Experimental band gaps (Zhuo et al.); composition-only, so X_base == X_full.
    df = load_dataframe_from_json(str(DATA_DIR / "expt_gap.json")).reset_index(drop=True)
    y = num(df["gap expt"]).to_numpy()
    X = magpie(df["formula"]).astype(float)
    base_cfg = dict(n_estimators=300, max_depth=6, learning_rate=0.05)
    return "Expt gap (experimental)", X, X, y, base_cfg


def metrics(pred, y_te):
    err = np.abs(pred - y_te)
    nz = y_te > 0
    return {
        "overall_acc": np.mean(err < TOL),
        "nz_acc_007": np.mean(err[nz] < TOL),
        "nz_acc_030": np.mean(err[nz] < 0.30),
        "nz_mae": np.mean(err[nz]),
        "nz_rmse": np.sqrt(np.mean(err[nz] ** 2)),
    }


def run_dataset(name, X_base, X_full, y, base_cfg):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    base_acc, imp_acc, recalls = [], [], []
    for tr, te in kf.split(X_full):
        y_tr, y_te = y[tr], y[te]
        ybin = (y > 0).astype(int)

        # Baseline: single-stage regressor on all data
        bm = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, **base_cfg)
        bm.fit(X_base.iloc[tr], y_tr)
        base_acc.append(metrics(bm.predict(X_base.iloc[te]), y_te))

        # Two-stage: classifier gate + non-zero log regressor + Magpie
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(ybin[tr] == 0).sum() / max((ybin[tr] == 1).sum(), 1),
            eval_metric="logloss", random_state=SEED)
        clf.fit(X_full.iloc[tr], ybin[tr])
        nz_tr = tr[y[tr] > 0]
        reg = xgb.XGBRegressor(
            n_estimators=700, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            reg_lambda=1.5, random_state=SEED)
        reg.fit(X_full.iloc[nz_tr], np.log1p(y[nz_tr]))
        gap_hat = np.clip(np.expm1(reg.predict(X_full.iloc[te])), 0, None)
        is_nm = clf.predict_proba(X_full.iloc[te])[:, 1] >= NONMETAL_THRESHOLD
        imp_acc.append(metrics(is_nm * gap_hat, y_te))
        recalls.append(np.mean(is_nm[y_te > 0] == 1))

    def agg(rows, key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std()

    print(f"\n=== {name}  ({N_SPLITS}-fold CV, mean +/- std) ===")
    print(f"{'metric':<22}{'baseline':>18}{'two-stage':>18}")
    for key, label in [("nz_mae", "non-zero MAE (eV)"),
                       ("nz_rmse", "non-zero RMSE (eV)"),
                       ("nz_acc_007", "non-zero acc<0.07"),
                       ("nz_acc_030", "non-zero acc<0.30"),
                       ("overall_acc", "overall acc<0.07")]:
        bm_, bs_ = agg(base_acc, key)
        im_, is_ = agg(imp_acc, key)
        print(f"{label:<22}{bm_:>8.3f} +/-{bs_:<6.3f}{im_:>8.3f} +/-{is_:<6.3f}")
    rm, rs = np.mean(recalls), np.std(recalls)
    print(f"{'classifier recall':<22}{'-':>16}{rm:>9.3f} +/-{rs:<6.3f}")


if __name__ == "__main__":
    for loader in (load_wolverton, load_castelli, load_expt_gap):
        run_dataset(*loader())
