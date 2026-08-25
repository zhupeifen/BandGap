#!/bin/bash
# SLURM batch script for the structure-based GNN on mp_gap.
# Submit with:  sbatch scripts/run_gnn_hpc.slurm
# Adjust the partition / module / account names to YOUR cluster (lines marked <--).

#SBATCH --job-name=gnn_mpgap
#SBATCH --partition=gpu            # <-- your GPU partition name
#SBATCH --gres=gpu:1               # 1 GPU is enough
#SBATCH --cpus-per-task=8          # for graph building / data loading
#SBATCH --mem=48G                  # full mp_gap.json load needs ample RAM
#SBATCH --time=06:00:00
#SBATCH --output=gnn_mpgap_%j.out
# #SBATCH --account=YOUR_ALLOCATION  # <-- uncomment if your cluster requires it

set -e
module purge
module load python/3.11 cuda/12.4   # <-- match your cluster's module names

# ----- one-time environment setup (run these on a LOGIN node, which has internet) -----
#   python -m venv ~/gnnenv
#   source ~/gnnenv/bin/activate
#   pip install torch --index-url https://download.pytorch.org/whl/cu124   # match CUDA module
#   pip install -r scripts/requirements_gnn.txt
# --------------------------------------------------------------------------------------

source ~/gnnenv/bin/activate

cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"; python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# full run on all non-zero mp_gap structures (needs data/mp_gap.json transferred):
python scripts/gnn_mp_gap.py --full --epochs 200

# (or a fast check on the cached 8k subset that ships in data/:)
# python scripts/gnn_mp_gap.py --n 8000 --epochs 200
