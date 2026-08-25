"""
Train a MEGNet-style graph network (Chen et al., Chem. Mater. 2019) on the same
pre-built mp_gap crystal graphs used for CGCNN, so the only thing that changes is
the architecture. MEGNet differs from CGCNN by carrying a *global* state vector
that is updated each block from the node and edge sets and fed back into both, a
node/edge/global message-passing scheme. Self-contained: needs only torch,
torch_geometric, xgboost, scikit-learn, numpy. Uses GPU automatically if present.

    python megnet_train_hpc.py --data _mp_gap_gnn_ready_full.pkl --epochs 200
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter
from sklearn.model_selection import train_test_split

SEED = 42
CUTOFF = 5.0
GAUSS = np.linspace(0, CUTOFF, 41).astype("float32")
GW = float(GAUSS[1] - GAUSS[0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
if "SLURM_CPUS_PER_TASK" in os.environ:
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))


def build(g, y):
    ea = np.exp(-((g["dist"][:, None] - GAUSS[None, :]) ** 2) / GW ** 2).astype("float32")
    return Data(x=torch.tensor(g["z"].astype("int64")),
                edge_index=torch.tensor(g["edge_index"].astype("int64")),
                edge_attr=torch.tensor(ea),
                y=torch.tensor([np.log1p(y)], dtype=torch.float))


def mlp(inp, out):
    return nn.Sequential(nn.Linear(inp, out), nn.Softplus(),
                         nn.Linear(out, out), nn.Softplus())


class MEGNetBlock(nn.Module):
    """One MEGNet update: edges <- (v_i, v_j, e, u); nodes <- (v, agg_e, u);
    global <- (mean_v, mean_e, u). Residual on each, BatchNorm for stability."""
    def __init__(self, dim=64):
        super().__init__()
        self.phi_e = mlp(4 * dim, dim)
        self.phi_v = mlp(3 * dim, dim)
        self.phi_u = mlp(3 * dim, dim)
        self.bn_e = nn.BatchNorm1d(dim)
        self.bn_v = nn.BatchNorm1d(dim)

    def forward(self, v, e, u, edge_index, ebatch, nbatch):
        src, dst = edge_index
        e2 = e + self.bn_e(self.phi_e(torch.cat([v[src], v[dst], e, u[ebatch]], dim=-1)))
        agg_e = scatter(e2, dst, dim=0, dim_size=v.size(0), reduce="mean")
        v2 = v + self.bn_v(self.phi_v(torch.cat([v, agg_e, u[nbatch]], dim=-1)))
        ue = scatter(e2, ebatch, dim=0, dim_size=u.size(0), reduce="mean")
        uv = scatter(v2, nbatch, dim=0, dim_size=u.size(0), reduce="mean")
        u2 = u + self.phi_u(torch.cat([uv, ue, u], dim=-1))
        return v2, e2, u2


class MEGNet(nn.Module):
    def __init__(self, dim=64, edim=41, n_blocks=3):
        super().__init__()
        self.embed = nn.Embedding(100, dim)
        self.e0 = nn.Linear(edim, dim)
        self.blocks = nn.ModuleList([MEGNetBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(3 * dim, dim), nn.Softplus(),
                                  nn.Linear(dim, 1))

    def forward(self, d):
        v = self.embed(d.x)
        e = self.e0(d.edge_attr)
        nbatch = d.batch
        ebatch = nbatch[d.edge_index[0]]
        n_graphs = int(nbatch.max()) + 1
        u = torch.zeros(n_graphs, v.size(-1), device=v.device)
        for blk in self.blocks:
            v, e, u = blk(v, e, u, d.edge_index, ebatch, nbatch)
        vp = global_mean_pool(v, nbatch)
        ep = scatter(e, ebatch, dim=0, dim_size=n_graphs, reduce="mean")
        return self.head(torch.cat([vp, ep, u], dim=-1)).squeeze(-1)


def mae_eval(model, loader):
    model.eval(); err = []
    with torch.no_grad():
        for d in loader:
            d = d.to(DEVICE)
            err.append((torch.clamp(torch.expm1(model(d)), min=0) - torch.expm1(d.y)).abs().cpu())
    return torch.cat(err).mean().item()


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="_mp_gap_gnn_ready_full.pkl")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="split + init seed (R2: run seeds 0-4; 42 = original)")
    args = ap.parse_args()
    SEED = args.seed
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"seed: {SEED}")
    print(f"device: {DEVICE}, threads: {torch.get_num_threads()}")

    d = pickle.load(open(args.data, "rb"))
    y, X = d["y"], d["X"]
    t = time.time()
    graphs = [build(g, yi) for g, yi in zip(d["graphs"], y)]
    print(f"materials: {len(graphs)}; graphs built in {time.time() - t:.0f}s")

    tr, te = train_test_split(range(len(graphs)), test_size=0.2, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1, random_state=SEED)
    tl = DataLoader([graphs[i] for i in tr], batch_size=128, shuffle=True)
    vl = DataLoader([graphs[i] for i in va], batch_size=256)
    el = DataLoader([graphs[i] for i in te], batch_size=256)

    model = MEGNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            b = b.to(DEVICE); opt.zero_grad()
            F.mse_loss(model(b), b.y).backward(); opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            v = mae_eval(model, vl)
            if v < best:
                best, best_state = v, {k: w.cpu().clone() for k, w in model.state_dict().items()}
            print(f"  epoch {ep:3d}  val non-zero MAE {v:.3f} eV", flush=True)
    model.load_state_dict(best_state)
    gnn_mae = mae_eval(model, el)

    import xgboost as xgb
    reg = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                           reg_lambda=1.5, random_state=SEED)
    reg.fit(X[tr], np.log1p(y[tr]))
    xgb_pred = np.clip(np.expm1(reg.predict(X[te])), 0, None)
    xgb_mae = float(np.mean(np.abs(xgb_pred - y[te])))

    print(f"\n=== non-zero MAE on held-out test (eV), mp_gap ({len(graphs)} materials) ===")
    print(f"  XGBoost (Magpie, composition-only): {xgb_mae:.3f}")
    print(f"  MEGNet (crystal structure)        : {gnn_mae:.3f}")
    print(f"  improvement: {100 * (xgb_mae - gnn_mae) / xgb_mae:+.1f}%")
    import json
    out = f"gnn_results_megnet_seed{SEED}.json"
    json.dump({"model": "MEGNet", "seed": SEED, "n_materials": len(graphs),
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "epochs": args.epochs, "xgb_mae": xgb_mae, "gnn_mae": float(gnn_mae),
               "reduction_pct": float(100 * (xgb_mae - gnn_mae) / xgb_mae)},
              open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
