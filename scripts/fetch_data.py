"""
Download the five public matminer datasets used in the paper into data/ as JSON
(the format the analysis scripts read with matminer.utils.io.load_dataframe_from_json).

    .venv/Scripts/python.exe scripts/fetch_data.py            # all five
    .venv/Scripts/python.exe scripts/fetch_data.py expt_gap   # one

The sixth dataset (tolerance-factor-screened perovskites, Tol_screened_ensemble_final.csv)
is third-party CC-BY 4.0 data from the Materials Data Facility, DOI 10.18126/dp3z-bp06
(Biswas & Mannodi-Kanakkithodi, 2024); download it from that record into data/.
mp_gap is ~675 MB (it ships pymatgen structures).
"""
import sys
from pathlib import Path

from matminer.datasets import load_dataset
from matminer.utils.io import store_dataframe_as_json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = ["expt_gap", "wolverton_oxides", "castelli_perovskites",
            "double_perovskites_gap", "mp_gap"]


def main(names):
    DATA_DIR.mkdir(exist_ok=True)
    for name in names:
        out = DATA_DIR / f"{name}.json"
        if out.exists():
            print(f"{out.name}: already present, skipping")
            continue
        print(f"downloading {name} ...", flush=True)
        df = load_dataset(name)
        store_dataframe_as_json(df, str(out))
        print(f"  wrote {out} ({len(df)} rows)")
        if name == "double_perovskites_gap":   # the analysis scripts read this one as CSV
            csv = DATA_DIR / "Dataset_double_perovskites_gap_v1.csv"
            df.to_csv(csv, index=False)
            print(f"  wrote {csv}")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
