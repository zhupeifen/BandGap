#!/bin/bash
# Create a CUDA GPU conda env on ORCA (torch cu121 for the T4 / driver 575 / CUDA 12.9).
set -e
CONDA=~/miniconda3/bin/conda
ENV=~/miniconda3/envs/gnngpu
PY=$ENV/bin/python

if [ -x "$PY" ] && $PY -c "import torch,torch_geometric,xgboost,sklearn" 2>/dev/null; then
    echo "GPU env already complete"; $PY -c "import torch; print('torch', torch.__version__)"; exit 0
fi
if [ ! -x "$PY" ]; then
    $CONDA create -y -n gnngpu --override-channels -c conda-forge python=3.10 pip
fi
$PY -m ensurepip --upgrade 2>/dev/null || true
$PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install -q torch_geometric xgboost scikit-learn numpy
$PY -c "import torch,torch_geometric,xgboost; print('GPU_ENV_OK torch', torch.__version__, 'pyg', torch_geometric.__version__)"
