"""
Materials Project (mp_gap) analysis: large-scale validation of the findings.

mp_gap has 106,113 entries (43.5% zero gaps), as pymatgen Structure objects with
no formula column. We derive composition from each structure and use Magpie
features (composition-only, like expt_gap but ~17x larger). Features are cached to
disk so re-runs are fast. Reports: zero-inflation, single- vs two-stage non-zero
metrics (5-fold CV), and random vs grouped-split generalization.

    .venv/Scripts/python.exe scripts/mp_gap_analysis.py
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, ShuffleSplit, GroupShuffleSplit
from matminer.utils.io import load_dataframe_from_json
from matminer.featurizers.composition import ElementProperty

from cv_evaluation import metrics, DATA_DIR, NONMETAL_THRESHOLD, SEED

warnings.filterwarnings("ignore")

CACHE = DATA_DIR / "_mp_gap_features.pkl"   # gitignored (under data/)
N_SPLITS = 5
_EP = ElementProperty.from_preset("magpie")


def build_cache():
    t = time.time()
    print("Loading mp_gap.json (large, ~80s)...")
    df = load_dataframe_from_json(str(DATA_DIR / "mp_gap.json")).reset_index(drop=True)
    y = pd.to_numeric(df["gap pbe"], errors="coerce").to_numpy()
    print(f"  loaded {len(df)} rows in {time.time()-t:.0f}s; featurizing compositions...")
    comps = [s.composition for s in df["structure"]]
    rows, chem, redform = [], [], []
    labels = _EP.feature_labels()
    for c in comps:
        try:
            rows.append(_EP.featurize(c))
        except Exception:
            rows.append([np.nan] * len(labels))
        chem.append("-".join(sorted(e.symbol for e in c.elements)))
        redform.append(c.reduced_formula)
    X = pd.DataFrame(rows, columns=labels).astype(float)
    pd.to_pickle({"X": X, "y": y, "chem": np.array(chem),
                  "redform": np.array(redform)}, CACHE)
    print(f"  cached features to {CACHE.name} ({time.time()-t:.0f}s total)")
    return X, y, np.array(chem), np.array(redform)


def load_cache():
    if CACHE.exists():
        print(f"Using cached features ({CACHE.name})")
        d = pd.read_pickle(CACHE)
        return d["X"], d["y"], d["chem"], d["redform"]
    return build_cache()


def two_stage(X, y, tr, te):
    ybin = (y > 0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=(ybin[tr] == 0).sum() / max(ybin[tr].sum(), 1),
                            eval_metric="logloss", random_state=SEED, n_jobs=-1)
    clf.fit(X.iloc[tr], ybin[tr])
    nz = tr[y[tr] > 0]
    reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                           reg_lambda=1.5, random_state=SEED, n_jobs=-1)
    reg.fit(X.iloc[nz], np.log1p(y[nz]))
    gap_hat = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
    is_nm = clf.predict_proba(X.iloc[te])[:, 1] >= NONMETAL_THRESHOLD
    return metrics(is_nm * gap_hat, y[te]), np.mean(is_nm[y[te] > 0] == 1)


def single_stage(X, y, tr, te):
    m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    m.fit(X.iloc[tr], y[tr])
    return metrics(m.predict(X.iloc[te]), y[te])


def main():
    X, y, chem, redform = load_cache()
    print(f"\nmp_gap: {len(y)} materials, zero gaps {int((y==0).sum())} "
          f"({(y==0).mean():.1%}), non-zero median {np.median(y[y>0]):.2f} eV")

    # 5-fold CV: single-stage vs two-stage (both Magpie -> pure method effect)
    kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    base, two, rec = [], [], []
    for tr, te in kf.split(X):
        base.append(single_stage(X, y, tr, te))
        m, r = two_stage(X, y, tr, te)
        two.append(m); rec.append(r)

    def ag(rows, k):
        v = np.array([r[k] for r in rows]); return v.mean(), v.std()
    print(f"\n=== mp_gap (5-fold CV, mean +/- std) ===")
    print(f"{'metric':<20}{'single-stage':>18}{'two-stage':>18}")
    for k, lab in [("nz_mae", "non-zero MAE"), ("nz_acc_007", "non-zero acc<0.07"),
                   ("nz_acc_030", "non-zero acc<0.30"), ("overall_acc", "overall acc<0.07")]:
        bm, bs = ag(base, k); tm, ts = ag(two, k)
        print(f"  {lab:<18}{bm:>8.3f} +/-{bs:<5.3f}{tm:>8.3f} +/-{ts:<5.3f}")
    print(f"  classifier recall (non-zero): {np.mean(rec):.3f} +/- {np.std(rec):.3f}")

    # Generalization: single split per strategy (large data; cheap & sufficient)
    print(f"\n=== mp_gap generalization (two-stage, single 80/20 split, non-zero MAE) ===")
    idx = np.arange(len(y))
    strat = [("random", next(ShuffleSplit(1, test_size=0.2, random_state=SEED).split(idx))),
             ("by-composition", next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(idx, groups=redform))),
             ("by-chemistry", next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(idx, groups=chem)))]
    rand = None
    for label, (tr, te) in strat:
        m, _ = two_stage(X, y, tr, te)
        if rand is None:
            rand = m["nz_mae"]
        print(f"  {label:<16} MAE {m['nz_mae']:.3f}   ({m['nz_mae']/rand:.2f}x random)")


if __name__ == "__main__":
    main()
