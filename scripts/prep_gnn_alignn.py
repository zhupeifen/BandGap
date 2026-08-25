"""
Re-build the full mp_gap crystal graphs WITH bond displacement vectors, so an
ALIGNN-style line-graph network can compute bond angles on the cluster. Reuses the
Magpie features already computed in _mp_gap_gnn_ready_full.pkl (same deterministic
material order) to avoid re-featurizing ~60k compositions.

    .venv/Scripts/python.exe scripts/prep_gnn_alignn.py
"""
import pickle
import time
import warnings

import numpy as np
from matminer.utils.io import load_dataframe_from_json

warnings.filterwarnings("ignore")
CUTOFF = 5.0

print("loading mp_gap.json ...")
df = load_dataframe_from_json("data/mp_gap.json").reset_index(drop=True)
df = df.rename(columns={"gap pbe": "gap"})
df = df[df["gap"] > 0].reset_index(drop=True)
y = df["gap"].to_numpy(dtype=np.float32)
structs = df["structure"].tolist()
del df

# Reuse the Magpie matrix from the existing full pickle (identical material order).
ref = pickle.load(open("data/_mp_gap_gnn_ready_full.pkl", "rb"))
X = ref["X"]
assert len(X) == len(structs) == len(ref["y"]), (len(X), len(structs), len(ref["y"]))
assert np.allclose(ref["y"], y, atol=1e-4), "material order mismatch vs reference pickle"
del ref

graphs = []
t = time.time()
for i, s in enumerate(structs):
    z = np.array([site.specie.Z for site in s], dtype=np.int32)
    c, nbr, img, dist = s.get_neighbor_list(CUTOFF)
    if len(c) == 0:
        c = np.arange(len(z)); nbr = np.arange(len(z))
        img = np.zeros((len(z), 3)); dist = np.full(len(z), 0.1)
    # actual bond vector: r(nbr) + image . lattice - r(center)
    vec = (s.cart_coords[nbr] + img @ s.lattice.matrix - s.cart_coords[c]).astype(np.float32)
    graphs.append({"z": z,
                   "edge_index": np.vstack([c, nbr]).astype(np.int32),
                   "dist": dist.astype(np.float32),
                   "vec": vec})
    if i and i % 10000 == 0:
        print(f"  {i} ... {time.time() - t:.0f}s")

out = "data/_mp_gap_alignn_full.pkl"
with open(out, "wb") as f:
    pickle.dump({"graphs": graphs, "X": X.astype(np.float32), "y": y}, f, protocol=4)
print(f"done: {len(graphs)} graphs in {time.time() - t:.0f}s; saved {out}")
