# R2 revision: check the five-seed GNN jobs on ORCA and pull their results.
# Login uses the SSH key configured as Host "medea" in ~/.ssh/config (no password).
#   cd "<repo>\scripts"; .\hpc_pull_seed_results.ps1
# Jobs submitted 2026-08-25: 300506 cgcnn_seeds, 300507 megnet_seeds, 300508 alignn_seeds (dir ~/bandgap)

ssh medea "squeue -u pzhu -o '%.10i %.14j %.2t %.10M %R'; cd ~/bandgap; ls -la gnn_results_*_seed*.json 2>/dev/null; grep -h 'improvement' *_seed?.log 2>/dev/null"
scp "medea:~/bandgap/gnn_results_*_seed*.json" .
scp "medea:~/bandgap/*_seeds.*.out" "medea:~/bandgap/*_seed?.log" ..\data\gnn_seed_logs\ 2>$null
python summarize_gnn_seeds.py .
