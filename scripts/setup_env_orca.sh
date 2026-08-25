#!/bin/bash
# Create the GNN conda env on ORCA from conda-forge (avoids Anaconda channel ToS),
# then pip-install the minimal stack. Idempotent.
set -e
CONDA=~/miniconda3/bin/conda
ENV=~/miniconda3/envs/gnn
PY=$ENV/bin/python

if [ -x "$PY" ] && $PY -c "import torch,torch_geometric,xgboost,sklearn" 2>/dev/null; then
    echo "ENV already complete"; $PY -c "import torch,torch_geometric; print('torch',torch.__version__,'pyg',torch_geometric.__version__)"; exit 0
fi

if [ ! -x "$PY" ]; then
    $CONDA create -y -n gnn --override-channels -c conda-forge python=3.10 pip
fi
$PY -m ensurepip --upgrade 2>/dev/null || true
$PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -q torch_geometric xgboost scikit-learn numpy
$PY -c "import torch,torch_geometric,xgboost,sklearn; print('ENV_OK torch', torch.__version__, 'pyg', torch_geometric.__version__)"
