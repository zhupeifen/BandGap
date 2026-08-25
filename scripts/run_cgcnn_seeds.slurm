#!/bin/bash -l
#SBATCH -J cgcnn_seeds
#SBATCH -N 1
#SBATCH -p t4
#SBATCH -t 24:00:00
#SBATCH -o cgcnn_seeds.%j.out
#SBATCH -e cgcnn_seeds.%j.err
#SBATCH --mail-user=pzhu@missouri.edu
#SBATCH --mail-type=END,FAIL

# R2 revision (Reviewer 1, point 2c): five independent split/init seeds of the
# structure-vs-composition comparison. The t4 node is allocated whole (4 x T4, 250 GB),
# so the five seeds run concurrently, seed k on GPU k % 4. Each seed writes
# gnn_results_cgcnn_seed<k>.json and its own log cgcnn_seed<k>.log.
cd "$SLURM_SUBMIT_DIR"
echo "node $(hostname)"; nvidia-smi -L
export OMP_NUM_THREADS=12
for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((k % 4)) ~/miniconda3/envs/gnngpu/bin/python gnn_train_hpc.py       --data _mp_gap_gnn_ready_full.pkl --epochs 200 --seed $k > cgcnn_seed$k.log 2>&1 &
done
wait
echo "all seeds finished"; ls -la gnn_results_cgcnn_seed*.json
