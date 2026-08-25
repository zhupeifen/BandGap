#!/bin/bash -l
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p bw
#SBATCH -t 12:00:00
#SBATCH -o gnn.%j.out
#SBATCH -e gnn.%j.err
#SBATCH --mail-user=pzhu@missouri.edu
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
export SLURM_CPUS_PER_TASK=$(nproc)
export OMP_NUM_THREADS=$(nproc)
echo "node $(hostname), cores $(nproc)"
~/miniconda3/envs/gnn/bin/python gnn_train_hpc.py --epochs 200
