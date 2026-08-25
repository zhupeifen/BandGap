#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p bw
#SBATCH -t 12:00:00
#SBATCH -c 16
#SBATCH -o crabnet_expt.%j.out
#SBATCH -e crabnet_expt.%j.err
#SBATCH --mail-user=pzhu@missouri.edu
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "node $(hostname); threads $OMP_NUM_THREADS"
~/miniconda3/envs/crab/bin/python crabnet_compare.py --csvdir crabnet_csv --epochs 300 --only expt_gap
