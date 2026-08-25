"""
Pre-build the crystal graphs and Magpie features for the mp_gap 8k subset into ONE
portable file, so the HPC training script needs only torch + torch_geometric
(no pymatgen / matminer build on the cluster). Run locally where pymatgen and
matminer are installed.

    .venv/Scripts/python.exe scripts/prep_gnn_data.py
"""

import argparse
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty

warnings.filterwarnings("ignore")
CUTOFF = 5.0          # Gaussian distance expansion is done on the cluster (compact storage)

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="all non-zero mp_gap (~60k) instead of the 8k subset")
args = ap.parse_args()

if args.full:
    from matminer.utils.io import load_dataframe_from_json
    df = load_dataframe_from_json("data/mp_gap.json").reset_index(drop=True)
    df = df.rename(columns={"gap pbe": "gap"})
    df = df[df["gap"] > 0].reset_index(drop=True)
    out = "data/_mp_gap_gnn_ready_full.pkl"
else:
    df = pd.read_pickle("data/_mp_gap_subset8k.pkl")
    out = "data/_mp_gap_gnn_ready.pkl"
print(f"non-zero materials: {len(df)} -> {out}")

ep = ElementProperty.from_preset("magpie")
y = df["gap"].to_numpy(dtype=np.float32)
structs = df["structure"].tolist()
del df                                    # free the big frame before building graphs

graphs, X = [], []
t = time.time()
for i, s in enumerate(structs):
    z = np.array([site.specie.Z for site in s], dtype=np.int32)
    c, nbr, _img, dist = s.get_neighbor_list(CUTOFF)
    if len(c) == 0:
        c = np.arange(len(z)); nbr = np.arange(len(z)); dist = np.full(len(z), 0.1)
    graphs.append({"z": z,
                   "edge_index": np.vstack([c, nbr]).astype(np.int32),
                   "dist": dist.astype(np.float32)})
    X.append(ep.featurize(s.composition))
    if i and i % 10000 == 0:
        print(f"  {i} ... {time.time() - t:.0f}s")

data = {"graphs": graphs, "X": np.array(X, dtype=np.float32), "y": y}
with open(out, "wb") as f:
    pickle.dump(data, f, protocol=4)
print(f"done: {len(graphs)} graphs in {time.time() - t:.0f}s; X {data['X'].shape}; saved {out}")
