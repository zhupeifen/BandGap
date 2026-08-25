#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p t4
#SBATCH -t 12:00:00
#SBATCH -o gnngpu.%j.out
#SBATCH -e gnngpu.%j.err
#SBATCH --mail-user=pzhu@missouri.edu
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
echo "node $(hostname)"; nvidia-smi -L 2>/dev/null | head -1
~/miniconda3/envs/gnngpu/bin/python gnn_train_hpc.py --data _mp_gap_gnn_ready_full.pkl --epochs 200
