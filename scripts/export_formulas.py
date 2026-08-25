"""
Export formula + band-gap CSVs (non-zero materials only) for the three CV datasets,
so a composition-only deep net (CrabNet) can be trained on the cluster without
pymatgen/matminer. One row per material: columns formula, target.

    .venv/Scripts/python.exe scripts/export_formulas.py
"""
from pathlib import Path
import pandas as pd
from matminer.utils.io import load_dataframe_from_json

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "crabnet_csv"
OUT.mkdir(exist_ok=True)

SETS = [
    ("wolverton", "wolverton_oxides.json", "gap pbe"),
    ("castelli", "castelli_perovskites.json", "gap gllbsc"),
    ("expt_gap", "expt_gap.json", "gap expt"),
]

for name, fn, gapcol in SETS:
    df = load_dataframe_from_json(str(DATA / fn)).reset_index(drop=True)
    g = pd.to_numeric(df[gapcol], errors="coerce")
    out = pd.DataFrame({"formula": df["formula"].astype(str), "target": g})
    out = out[out["target"] > 0].dropna().reset_index(drop=True)
    out.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(out)} non-zero rows -> {OUT / (name + '.csv')}")
