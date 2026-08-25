"""
Aggregate the five-seed structure-vs-composition runs (R2 revision, Reviewer 1 point 2c).

Reads gnn_results_{cgcnn,megnet,alignn}_seed{k}.json written by the *_train_hpc.py
scripts (pull them from ORCA with pscp first) and prints Table S3 as mean +/- SD over
seeds, the per-seed paired reduction vs the composition-only XGBoost baseline trained on
the identical split, and whether every seed favours the graph model.

    python scripts/summarize_gnn_seeds.py [dir-with-json]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
rows = {}
for f in sorted(glob.glob(str(d / "gnn_results_*_seed*.json"))):
    r = json.load(open(f))
    rows.setdefault(r["model"], []).append(r)
if not rows:
    sys.exit(f"no gnn_results_*_seed*.json in {d}")

print(f"{'model':<10}{'seeds':>6}{'GNN MAE (eV)':>18}{'XGB MAE (eV)':>18}{'reduction %':>22}{'all seeds better':>18}")
out = {}
for model, rs in rows.items():
    rs = sorted(rs, key=lambda r: r["seed"])
    g = np.array([r["gnn_mae"] for r in rs]); x = np.array([r["xgb_mae"] for r in rs])
    red = 100 * (x - g) / x
    print(f"{model:<10}{len(rs):>6}{g.mean():>11.3f} ± {g.std(ddof=1):<5.3f}"
          f"{x.mean():>11.3f} ± {x.std(ddof=1):<5.3f}"
          f"{red.mean():>10.1f} ({red.min():.1f}–{red.max():.1f}){str(bool((g < x).all())):>18}")
    out[model] = {"seeds": [r["seed"] for r in rs], "gnn_mae": g.tolist(), "xgb_mae": x.tolist(),
                  "gnn_mean": g.mean(), "gnn_sd": g.std(ddof=1), "xgb_mean": x.mean(),
                  "xgb_sd": x.std(ddof=1), "reduction_mean": red.mean(),
                  "reduction_min": red.min(), "reduction_max": red.max()}
(d / "gnn_seed_summary.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {d / 'gnn_seed_summary.json'}  (SD uses ddof=1; XGB baseline is per-split, so it varies too)")
