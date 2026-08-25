"""
Train an ALIGNN-style line-graph network (after Choudhary & DeCost, npj Comput.
Mater. 2021) on the mp_gap crystal graphs. Each ALIGNN layer alternates an
EdgeGatedGraphConv on the bond *line graph* (nodes = bonds, edges = bond pairs
sharing an atom, features = bond angles) with an EdgeGatedGraphConv on the atom
graph (nodes = atoms, edge features = the updated bond representations). This
injects bond-angle information that CGCNN/MEGNet (distance-only) ignore.
Self-contained: torch + torch_geometric + xgboost + sklearn + numpy.

    python alignn_train_hpc.py --data _mp_gap_alignn_full.pkl --epochs 200
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
GD = np.linspace(0, CUTOFF, 41).astype("float32")          # distance basis
GDW = float(GD[1] - GD[0])
GA = np.linspace(-1.0, 1.0, 9).astype("float32")           # cos(angle) basis
GAW = float(GA[1] - GA[0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
if "SLURM_CPUS_PER_TASK" in os.environ:
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))


class CData(Data):
    """Carries an atom graph and its bond line graph; line_index references bond
    ids, so it must increment by the bond count (not the atom count) on batching."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == "line_index":
            return self.edge_index.size(1)
        if key == "edge_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("edge_index", "line_index"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def cap_knn(ei, dist, vec, k=12):
    """Keep each atom's k nearest neighbours (ALIGNN's standard 12-NN graph).
    Bounds the line-graph size; the dense 5 A graph would otherwise explode."""
    src = ei[0]
    if src.shape[0] == 0:
        return ei, dist, vec
    order = np.lexsort((dist, src))                 # by src, then ascending distance
    _, counts = np.unique(src[order], return_counts=True)
    rank = np.arange(order.shape[0]) - np.repeat(np.cumsum(counts) - counts, counts)
    keep = np.sort(order[rank < k])
    return ei[:, keep], dist[keep], vec[keep]


def line_graph(edge_index):
    """Directed line-graph edges (e1, e2) where dst(e1) == src(e2): a path
    u->v->w sharing the middle atom v. Vectorised per structure."""
    src, dst = edge_index
    E = src.shape[0]
    N = int(max(src.max(), dst.max())) + 1 if E else 0
    counts = np.bincount(src, minlength=N)
    order = np.argsort(src, kind="stable")          # outgoing edges grouped by src
    ptr = np.concatenate([[0], np.cumsum(counts)])
    cnt = counts[dst]                                # out-degree of each edge's dst
    e1 = np.repeat(np.arange(E), cnt)
    rep_start = np.repeat(ptr[dst], cnt)
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    e2 = order[rep_start + within]
    return e1.astype(np.int64), e2.astype(np.int64)


def build(g, y, k=12):
    ei = g["edge_index"].astype(np.int64)
    dist = g["dist"].astype("float32")
    vec = g["vec"].astype("float32")
    ei, dist, vec = cap_knn(ei, dist, vec, k)
    e1, e2 = line_graph(ei)
    # angle at the shared atom: between (v->u) = -vec[e1] and (v->w) = vec[e2]
    a = -vec[e1]; b = vec[e2]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    edge_attr = np.exp(-((dist[:, None] - GD[None, :]) ** 2) / GDW ** 2).astype("float32")
    angle_attr = np.exp(-((cos[:, None] - GA[None, :]) ** 2) / GAW ** 2).astype("float32")
    return CData(x=torch.tensor(g["z"].astype("int64")),
                 edge_index=torch.tensor(ei),
                 edge_attr=torch.tensor(edge_attr),
                 line_index=torch.tensor(np.vstack([e1, e2])),
                 angle_attr=torch.tensor(angle_attr),
                 y=torch.tensor([np.log1p(y)], dtype=torch.float))


class EdgeGatedConv(nn.Module):
    """EdgeGatedGraphConv (Dwivedi et al.): updates node features h and edge
    features e together, with sigmoid edge gates normalising the aggregation."""
    def __init__(self, D):
        super().__init__()
        self.A = nn.Linear(D, D); self.B = nn.Linear(D, D); self.C = nn.Linear(D, D)
        self.U = nn.Linear(D, D); self.V = nn.Linear(D, D)
        self.bn_e = nn.BatchNorm1d(D); self.bn_h = nn.BatchNorm1d(D)

    def forward(self, h, edge_index, e):
        src, dst = edge_index
        e_new = e + F.relu(self.bn_e(self.A(h[src]) + self.B(h[dst]) + self.C(e)))
        eta = torch.sigmoid(e_new)
        num = scatter(eta * self.V(h[src]), dst, dim=0, dim_size=h.size(0), reduce="sum")
        den = scatter(eta, dst, dim=0, dim_size=h.size(0), reduce="sum") + 1e-6
        h_new = h + F.relu(self.bn_h(self.U(h) + num / den))
        return h_new, e_new


class ALIGNNLayer(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.line = EdgeGatedConv(D)
        self.atom = EdgeGatedConv(D)

    def forward(self, x, edge_index, e_bond, line_index, e_angle):
        e_bond, e_angle = self.line(e_bond, line_index, e_angle)   # bonds as nodes
        x, e_bond = self.atom(x, edge_index, e_bond)               # atoms as nodes
        return x, e_bond, e_angle


class ALIGNN(nn.Module):
    def __init__(self, D=64, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(100, D)
        self.bond0 = nn.Linear(41, D)
        self.angle0 = nn.Linear(9, D)
        self.layers = nn.ModuleList([ALIGNNLayer(D) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(D, D), nn.Softplus(), nn.Linear(D, 1))

    def forward(self, d):
        x = self.embed(d.x)
        e_bond = self.bond0(d.edge_attr)
        e_angle = self.angle0(d.angle_attr)
        for lyr in self.layers:
            x, e_bond, e_angle = lyr(x, d.edge_index, e_bond, d.line_index, e_angle)
        return self.head(global_mean_pool(x, d.batch)).squeeze(-1)


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
    ap.add_argument("--data", default="_mp_gap_alignn_full.pkl")
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
    print(f"materials: {len(graphs)}; graphs+linegraphs built in {time.time() - t:.0f}s")

    tr, te = train_test_split(range(len(graphs)), test_size=0.2, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1, random_state=SEED)
    tl = DataLoader([graphs[i] for i in tr], batch_size=64, shuffle=True)
    vl = DataLoader([graphs[i] for i in va], batch_size=128)
    el = DataLoader([graphs[i] for i in te], batch_size=128)

    model = ALIGNN().to(DEVICE)
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
    print(f"  ALIGNN (structure + bond angles)  : {gnn_mae:.3f}")
    print(f"  improvement: {100 * (xgb_mae - gnn_mae) / xgb_mae:+.1f}%")
    import json
    out = f"gnn_results_alignn_seed{SEED}.json"
    json.dump({"model": "ALIGNN", "seed": SEED, "n_materials": len(graphs),
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "epochs": args.epochs, "xgb_mae": xgb_mae, "gnn_mae": float(gnn_mae),
               "reduction_pct": float(100 * (xgb_mae - gnn_mae) / xgb_mae)},
              open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
