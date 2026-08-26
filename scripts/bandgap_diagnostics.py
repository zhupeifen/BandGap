"""
bandgap_diagnostics -- run the paper's reporting checklist on ANY band-gap dataset.

Given a table of (formula, gap) it reports the quantities this paper argues should
accompany every band-gap ML result, so a practitioner can spot inflated headline
numbers before trusting them:

  1. Zero inflation        -- non-zero fraction; why aggregate accuracy can mislead.
  2. Non-zero error        -- two-stage non-zero MAE/RMSE (5-fold CV), the honest metric.
  3. Gate quality          -- classifier AUC and the share of error from misgating.
  4. Fractional-formula share -- operational source-formula proxy.
  5. Split sensitivity        -- random vs chemistry-grouped non-zero MAE.

It is composition-only (Magpie features from the formula), so it works on any dataset
without native structural columns, and it reuses the exact model configuration from the
manuscript (cv_evaluation / cv_grouped_splits / gate_analysis).

    # CSV with a 'formula' and a 'gap' column:
    .venv/Scripts/python.exe scripts/bandgap_diagnostics.py --csv mydata.csv --gap-col gap
    # a matminer JSON the paper uses:
    .venv/Scripts/python.exe scripts/bandgap_diagnostics.py --json data/expt_gap.json --gap-col "gap expt"

    # or from Python:
    from bandgap_diagnostics import diagnose
    report = diagnose(df["formula"], df["gap"], name="mydata")
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from pymatgen.core import Composition

from cv_evaluation import magpie, metrics, NONMETAL_THRESHOLD, SEED
from cv_grouped_splits import two_stage
from gate_analysis import out_of_fold, decompose, TAU

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


def _is_fractional(comp):
    """Return whether the source formula contains a non-integer coefficient.

    This is an operational formula-level proxy, not a crystallographic test for
    site disorder.  Do not reduce the coefficient ratios to integers: doing so
    would incorrectly classify common 50/50 formulas such as
    CsPb0.5Sn0.5Br3 as non-fractional.
    """
    a = np.array(list(comp.get_el_amt_dict().values()))
    return not np.all(np.abs(a - np.round(a)) < 1e-3)


# Guard the definition used by the manuscript and command-line report.
assert _is_fractional(Composition("Hg0.7Cd0.3Te"))
assert _is_fractional(Composition("CsPb0.5Sn0.5Br3"))
assert not _is_fractional(Composition("CaTiO3"))


def composition_stats(formulas):
    """Parse formulas once: validity, Magpie features, group keys, solid-solution flag."""
    valid_idx, comp_key, chem_key, frac = [], [], [], []
    for i, f in enumerate(formulas):
        try:
            c = Composition(str(f))
        except Exception:
            continue
        valid_idx.append(i)
        comp_key.append(c.reduced_formula)
        chem_key.append("-".join(sorted(e.symbol for e in c.elements)))
        frac.append(_is_fractional(c))
    return (np.array(valid_idx), np.array(comp_key), np.array(chem_key),
            np.array(frac, dtype=bool))


def _single_stage_cv(X, y, n_splits, seed):
    """Non-zero MAE/RMSE for an all-non-zero dataset (no gate needed)."""
    maes, rmses = [], []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                               reg_lambda=1.5, random_state=seed)
        reg.fit(X.iloc[tr], np.log1p(y[tr]))
        pred = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
        m = metrics(pred, y[te])
        maes.append(m["nz_mae"]); rmses.append(m["nz_rmse"])
    return (float(np.mean(maes)), float(np.std(maes)),
            float(np.mean(rmses)), float(np.std(rmses)))


def _two_stage_cv(X, y, splits):
    rows = [two_stage(X, y, tr, te) for tr, te in splits]
    mae = np.array([r["nz_mae"] for r in rows])
    rmse = np.array([r["nz_rmse"] for r in rows])
    return mae.mean(), mae.std(), rmse.mean(), rmse.std()


def diagnose(formulas, gaps, name="dataset", do_grouped=True, do_gate=True,
             extra_features=None, n_splits=5, seed=SEED, verbose=True):
    """extra_features: optional DataFrame of native numeric columns (same length/order
    as `formulas`) concatenated to the Magpie features, e.g. a dataset's energetic or
    structural columns; omit it for a composition-only (universal) diagnostic."""
    formulas = pd.Series(list(formulas)).reset_index(drop=True)
    gaps = pd.to_numeric(pd.Series(list(gaps)).reset_index(drop=True), errors="coerce").to_numpy()
    if extra_features is not None:
        extra_features = pd.DataFrame(extra_features).reset_index(drop=True)

    vidx, comp_key, chem_key, frac = composition_stats(formulas)
    keep = vidx[np.isfinite(gaps[vidx])]
    # re-align everything to the kept rows
    sel = np.isin(vidx, keep)
    comp_key, chem_key, frac = comp_key[sel], chem_key[sel], frac[sel]
    fvalid = formulas.iloc[keep].reset_index(drop=True)
    y = gaps[keep].astype(float)
    feat_note = "Magpie" if extra_features is None else "Magpie + native columns"
    if verbose:
        print(f"  featurizing {len(fvalid)} formulas ({feat_note}) ...")
    X = magpie(fvalid).astype(float).reset_index(drop=True)
    if extra_features is not None:
        nat = extra_features.iloc[keep].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        X = pd.concat([nat, X], axis=1)

    n = len(y)
    nz = y > 0
    has_zeros = int((~nz).sum()) > 0
    rep = {
        "name": name, "n": int(n), "n_nonzero": int(nz.sum()),
        "nonzero_fraction": float(nz.mean()),
        "gap_min": float(y[nz].min()) if nz.any() else None,
        "gap_max": float(y[nz].max()) if nz.any() else None,
        "gap_median": float(np.median(y[nz])) if nz.any() else None,
        "solid_solution_share": float(frac.mean()),
        "composition_redundancy": float(1 - len(set(comp_key)) / n),
        "chemistry_redundancy": float(1 - len(set(chem_key)) / n),
        "features": feat_note,
    }

    # --- non-zero error (two-stage if zero-inflated, else single-stage) ---
    if has_zeros:
        splits = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
        mae_m, mae_s, rmse_m, rmse_s = _two_stage_cv(X, y, splits)
        rep["model"] = "two-stage (classify-then-regress)"
    else:
        mae_m, mae_s, rmse_m, rmse_s = _single_stage_cv(X, y, n_splits, seed)
        rep["model"] = "single-stage regressor (no zeros to gate)"
    rep.update(nz_mae=mae_m, nz_mae_std=mae_s, nz_rmse=rmse_m, nz_rmse_std=rmse_s)

    # --- gate diagnostics ---
    if has_zeros and do_gate:
        p, gap_hat, ybin = out_of_fold(X, y)
        reg_c, mis_c = decompose(p, gap_hat, y, TAU)
        rep["gate"] = {
            "auc": float(roc_auc_score(ybin, p)),
            "tau": TAU,
            "misgate_rate": float(np.mean(p[nz] < TAU)),
            "reg_component_eV": float(reg_c),
            "misgate_component_eV": float(mis_c),
            "misgate_fraction_of_error": float(mis_c / (reg_c + mis_c)) if (reg_c + mis_c) else 0.0,
        }

    # --- split optimism (random vs grouped) ---
    if do_grouped:
        ng = min(n_splits, len(set(comp_key)), len(set(chem_key)))
        if ng >= 2:
            ev = (_two_stage_cv if has_zeros else None)
            def grp_mae(splits):
                if has_zeros:
                    return _two_stage_cv(X, y, splits)[0]
                rows = []
                for tr, te in splits:
                    reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                                           subsample=0.8, colsample_bytree=0.8,
                                           min_child_weight=2, reg_lambda=1.5, random_state=seed)
                    reg.fit(X.iloc[tr], np.log1p(y[tr]))
                    pred = np.clip(np.expm1(reg.predict(X.iloc[te])), 0, None)
                    rows.append(metrics(pred, y[te])["nz_mae"])
                return float(np.mean(rows))
            rnd = grp_mae(list(KFold(ng, shuffle=True, random_state=seed).split(X)))
            comp = grp_mae(list(GroupKFold(ng).split(X, y, comp_key)))
            chem = grp_mae(list(GroupKFold(ng).split(X, y, chem_key)))
            rep["split_optimism"] = {
                "random_mae": float(rnd), "by_composition_mae": float(comp),
                "by_chemistry_mae": float(chem),
                "optimism_ratio": float(chem / rnd) if rnd else None,
                "n_groups_used": int(ng),
            }
        else:
            rep["split_optimism"] = {"note": "too few groups for a grouped split"}

    if verbose:
        _print_report(rep)
    return rep


def _print_report(r):
    line = "=" * 68
    print(f"\n{line}\n  BAND-GAP DIAGNOSTIC REPORT: {r['name']}\n{line}")
    print(f"  n = {r['n']}   non-zero = {r['n_nonzero']} "
          f"({100*r['nonzero_fraction']:.1f}%)   "
          f"gap {r['gap_min']:.2f}-{r['gap_max']:.2f} eV (median {r['gap_median']:.2f})"
          if r['gap_min'] is not None else f"  n = {r['n']}")
    print(f"  features: {r['features']}")

    print(f"\n  1. ZERO INFLATION")
    zi = 1 - r["nonzero_fraction"]
    print(f"     metals (gap = 0): {100*zi:.1f}%  -> aggregate accuracy is "
          f"{'dominated by metal ID; report non-zero error' if zi > 0.2 else 'less of a concern here'}.")

    print(f"\n  2. NON-ZERO ERROR ({r['model']}, {r.get('nz_mae_std') is not None and '5-fold CV' or ''})")
    print(f"     non-zero MAE  = {r['nz_mae']:.3f} +/- {r['nz_mae_std']:.3f} eV")
    print(f"     non-zero RMSE = {r['nz_rmse']:.3f} +/- {r['nz_rmse_std']:.3f} eV")

    if "gate" in r:
        g = r["gate"]
        print(f"\n  3. GATE QUALITY (metal/non-metal classifier, tau = {g['tau']})")
        print(f"     AUC = {g['auc']:.3f}   misgate rate (non-zero -> metal) = {100*g['misgate_rate']:.1f}%")
        print(f"     non-zero MAE = {g['reg_component_eV']:.3f} (regression) "
              f"+ {g['misgate_component_eV']:.3f} (misgating) eV; "
              f"misgating = {100*g['misgate_fraction_of_error']:.0f}% of error")
    else:
        print(f"\n  3. GATE QUALITY: n/a (no metals to gate)")

    print(f"\n  4. SOLID-SOLUTION SHARE")
    ss = r["solid_solution_share"]
    print(f"     fractional-formula entries: {100*ss:.1f}%  -> grouped/random ratio is "
          f"{'LIKELY (run a grouped split)' if ss > 0.05 else 'unlikely from this cause'}.")

    if "split_optimism" in r and "optimism_ratio" in r["split_optimism"]:
        s = r["split_optimism"]
        print(f"\n  5. SPLIT OPTIMISM (non-zero MAE, {s['n_groups_used']}-fold)")
        print(f"     random          : {s['random_mae']:.3f} eV")
        print(f"     by-composition  : {s['by_composition_mae']:.3f} eV")
        print(f"     by-chemistry    : {s['by_chemistry_mae']:.3f} eV")
        ratio = s["optimism_ratio"]
        flag = "INFLATED random split" if ratio and ratio > 1.15 else "little leakage"
        print(f"     grouped/random MAE ratio = {ratio:.2f}x  -> {flag}")
    elif "split_optimism" in r:
        print(f"\n  5. SPLIT SENSITIVITY: {r['split_optimism'].get('note', 'n/a')}")
    print(line)


def _load(args):
    if args.json:
        from matminer.utils.io import load_dataframe_from_json
        df = load_dataframe_from_json(args.json)
        src = args.json
    else:
        df = pd.read_csv(args.csv)
        src = args.csv
    if args.formula_col not in df.columns:
        raise SystemExit(f"formula column '{args.formula_col}' not in {list(df.columns)}")
    if args.gap_col not in df.columns:
        raise SystemExit(f"gap column '{args.gap_col}' not in {list(df.columns)}")
    extra = None
    if args.feature_cols:
        cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"feature columns not found: {missing}")
        extra = df[cols]
    name = args.name or Path(src).stem
    return df[args.formula_col], df[args.gap_col], name, extra


def main():
    ap = argparse.ArgumentParser(description="Band-gap dataset diagnostic (paper checklist).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="CSV file with formula and gap columns")
    g.add_argument("--json", help="matminer JSON dataframe")
    ap.add_argument("--formula-col", default="formula")
    ap.add_argument("--gap-col", required=True)
    ap.add_argument("--feature-cols", default=None,
                    help="comma-separated native numeric columns to add to Magpie "
                         "(e.g. 'e_form,e_hull'); omit for a composition-only diagnostic")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-grouped", action="store_true", help="skip the grouped-split probe")
    ap.add_argument("--no-gate", action="store_true", help="skip gate diagnostics")
    ap.add_argument("--out-json", default=None, help="write the report dict to this path")
    args = ap.parse_args()

    formulas, gaps, name, extra = _load(args)
    rep = diagnose(formulas, gaps, name=name, extra_features=extra,
                   do_grouped=not args.no_grouped, do_gate=not args.no_gate)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
