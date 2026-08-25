"""
Structure-based GNN (CGCNN) vs. composition-only XGBoost on Materials Project
band gaps (mp_gap). Tests whether reading the crystal structure beats the
composition-only Magpie features that bound accuracy in the main study.

Device-aware: uses CUDA automatically if available (recommended). On CPU it is
slow -- use --n to subsample. On a GPU machine, run the full set with --full.

    # quick smoke test (CPU ok):
    .venv/Scripts/python.exe scripts/gnn_mp_gap.py --n 1500 --epochs 5
    # full run (GPU strongly recommended):
    .venv/Scripts/python.exe scripts/gnn_mp_gap.py --full --epochs 200

Setup on a GPU machine (fresh venv):
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install torch_geometric pymatgen matminer xgboost scikit-learn pandas numpy
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
SEED = 42
CUTOFF = 5.0                     # neighbour cutoff (Angstrom)
GAUSS = np.linspace(0, CUTOFF, 41)   # distance expansion centres
GWIDTH = GAUSS[1] - GAUSS[0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- graph build
def structure_to_graph(struct, y):
    z = np.array([site.specie.Z for site in struct], dtype=np.int64)
    c, nbr, _img, dist = struct.get_neighbor_list(CUTOFF)   # i, j, image, distance
    if len(c) == 0:                                         # isolated -> self loop
        c = np.arange(len(z)); nbr = np.arange(len(z)); dist = np.full(len(z), 0.1)
    edge_index = torch.tensor(np.vstack([c, nbr]), dtype=torch.long)
    edge_attr = np.exp(-((dist[:, None] - GAUSS[None, :]) ** 2) / GWIDTH ** 2)
    return Data(x=torch.tensor(z), edge_index=edge_index,
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=torch.tensor([np.log1p(y)], dtype=torch.float))


# ------------------------------------------------------------------- CGCNN
class CGConv(MessagePassing):
    def __init__(self, dim, edim):
        super().__init__(aggr="add")
        self.lin = nn.Linear(2 * dim + edim, 2 * dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x + self.bn(out)

    def message(self, x_i, x_j, edge_attr):
        z = self.lin(torch.cat([x_i, x_j, edge_attr], dim=-1))
        f, s = z.chunk(2, dim=-1)
        return torch.sigmoid(f) * F.softplus(s)


class CGCNN(nn.Module):
    def __init__(self, dim=64, edim=41, n_conv=3):
        super().__init__()
        self.embed = nn.Embedding(100, dim)
        self.convs = nn.ModuleList([CGConv(dim, edim) for _ in range(n_conv)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.Softplus(),
                                  nn.Linear(dim, 1))

    def forward(self, d):
        x = self.embed(d.x)
        for conv in self.convs:
            x = conv(x, d.edge_index, d.edge_attr)
        return self.head(global_mean_pool(x, d.batch)).squeeze(-1)


def mae_eval(model, loader):
    model.eval(); err = []
    with torch.no_grad():
        for d in loader:
            d = d.to(DEVICE)
            pred = torch.clamp(torch.expm1(model(d)), min=0)
            err.append((pred - torch.expm1(d.y)).abs().cpu())
    return torch.cat(err).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="subsample size")
    ap.add_argument("--full", action="store_true", help="use all of mp_gap")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    # --- data: prefer the cached 8k subset, else load full mp_gap ---
    from pathlib import Path
    sub = Path(__file__).resolve().parent.parent / "data" / "_mp_gap_subset8k.pkl"
    if args.full or not sub.exists():
        from matminer.utils.io import load_dataframe_from_json
        df = load_dataframe_from_json(str(sub.parent / "mp_gap.json")).reset_index(drop=True)
        df = df.rename(columns={"gap pbe": "gap"})
        df = df[df["gap"] > 0].reset_index(drop=True)
    else:
        df = pd.read_pickle(sub)
    if not args.full and args.n < len(df):
        df = df.sample(n=args.n, random_state=SEED).reset_index(drop=True)
    print(f"materials (non-zero gap): {len(df)}")

    t = time.time()
    graphs = [structure_to_graph(s, g) for s, g in zip(df["structure"], df["gap"])]
    print(f"built {len(graphs)} graphs in {time.time() - t:.0f}s")

    tr, te = train_test_split(range(len(graphs)), test_size=0.2, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1, random_state=SEED)
    bs = 128
    tl = DataLoader([graphs[i] for i in tr], batch_size=bs, shuffle=True)
    vl = DataLoader([graphs[i] for i in va], batch_size=256)
    el = DataLoader([graphs[i] for i in te], batch_size=256)

    model = CGCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        for d in tl:
            d = d.to(DEVICE); opt.zero_grad()
            loss = F.mse_loss(model(d), d.y)
            loss.backward(); opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            vmae = mae_eval(model, vl)
            if vmae < best:
                best, best_state = vmae, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  epoch {ep:3d}  val non-zero MAE {vmae:.3f} eV")
    model.load_state_dict(best_state)
    gnn_mae = mae_eval(model, el)

    # --- XGBoost on Magpie composition features, same split ---
    from matminer.featurizers.composition import ElementProperty
    import xgboost as xgb
    ep_feat = ElementProperty.from_preset("magpie")
    X = np.array([ep_feat.featurize(s.composition) for s in df["structure"]])
    y = df["gap"].to_numpy()
    reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                           reg_lambda=1.5, random_state=SEED)
    reg.fit(X[tr], np.log1p(y[tr]))
    xgb_pred = np.clip(np.expm1(reg.predict(X[te])), 0, None)
    xgb_mae = float(np.mean(np.abs(xgb_pred - y[te])))

    print("\n=== non-zero MAE on held-out test (eV) ===")
    print(f"  XGBoost (Magpie, composition-only): {xgb_mae:.3f}")
    print(f"  CGCNN  (crystal structure)        : {gnn_mae:.3f}")
    print(f"  improvement: {100 * (xgb_mae - gnn_mae) / xgb_mae:+.1f}%")


if __name__ == "__main__":
    main()
