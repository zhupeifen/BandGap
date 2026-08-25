# Structure-based GNN on mp_gap (run on the GPU machine)

`scripts/gnn_mp_gap.py` trains a CGCNN on Materials Project crystal structures and
compares its non-zero band-gap MAE to composition-only XGBoost (Magpie) on the
same train/test split. It auto-uses CUDA if available.

The CPU smoke test (`--n 1500 --epochs 6`) only confirms the code runs; the GNN is
undertrained there. Run the real job on the GPU.

## Setup on the GPU computer (fresh venv)
```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
pip install torch_geometric pymatgen matminer xgboost scikit-learn pandas numpy
```

## Run
```
# fast, strong demo on the cached 8k-structure subset (data/_mp_gap_subset8k.pkl):
python scripts/gnn_mp_gap.py --n 8000 --epochs 200

# headline number on all non-zero mp_gap structures (needs data/mp_gap.json, 675 MB):
python scripts/gnn_mp_gap.py --full --epochs 200
```

Both the script and the cached 8k subset live in the OneDrive-synced project, so
they are already on the GPU machine. For `--full` make sure `data/mp_gap.json` has
finished syncing (it is large).

## What to expect / report
A well-trained CGCNN should beat composition-only XGBoost on these DFT gaps,
because it reads geometry the Magpie features discard. If it does, that is the
"new information, not a new learner" point from the paper: report the structure
GNN as the accuracy gain, while keeping the non-zero metric and grouped-split
evaluation unchanged.
