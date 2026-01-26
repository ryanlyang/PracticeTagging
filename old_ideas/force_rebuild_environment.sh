#!/bin/bash

# Force rebuild of corrupted conda environment
# Handles case where environment directory exists but is not a valid conda env

echo "=========================================="
echo "Force Rebuilding ATLAS KD Environment"
echo "=========================================="
echo ""

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh

# Deactivate any active environment
conda deactivate 2>/dev/null || true

# Get conda base path
CONDA_BASE=$(conda info --base)
ENV_PATH="$CONDA_BASE/envs/atlas_kd"

echo "Step 1: Forcefully removing corrupted environment directory..."
if [ -d "$ENV_PATH" ]; then
    echo "  Removing: $ENV_PATH"
    rm -rf "$ENV_PATH"
    echo "  ✓ Removed"
else
    echo "  Environment directory does not exist (OK)"
fi

echo ""
echo "Step 2: Creating fresh environment with Python 3.10..."
conda create -n atlas_kd python=3.10 -y

echo ""
echo "Step 3: Activating new environment..."
conda activate atlas_kd

# Verify we're in the right environment
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$CURRENT_ENV" != "atlas_kd" ]; then
    echo "ERROR: Failed to activate atlas_kd environment"
    echo "Current environment: $CURRENT_ENV"
    exit 1
fi

echo "  ✓ Activated atlas_kd"
echo "  Python: $(which python)"

echo ""
echo "Step 4: Installing PyTorch and core scientific packages via conda..."
conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision pytorch-cuda=11.8 \
    "numpy<2.0" \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn

echo ""
echo "Step 5: Installing remaining packages via pip..."
pip install h5py pandas energyflow tqdm tensorflow

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

# Test all critical imports
FAILED=0

python -c "import torch; print('✓ PyTorch:', torch.__version__)" || FAILED=1
python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available())" || FAILED=1
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" || FAILED=1
python -c "import scipy; print('✓ SciPy:', scipy.__version__)" || FAILED=1
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" || FAILED=1
python -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)" || FAILED=1
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)" || FAILED=1
python -c "import h5py; print('✓ h5py:', h5py.__version__)" || FAILED=1
python -c "import pandas; print('✓ pandas:', pandas.__version__)" || FAILED=1
python -c "import energyflow; print('✓ energyflow:', energyflow.__version__)" || FAILED=1

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=========================================="
    echo "✓ Environment rebuild complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Verify you're in the right directory:"
    echo "     cd /home/ryreu/atlas/PracticeTagging"
    echo ""
    echo "  2. Train shared models:"
    echo "     sbatch train_shared_models.sh"
    echo ""
    echo "  3. Monitor the job:"
    echo "     squeue -u \$USER"
    echo "     tail -f transformer_logs/train_shared_*.out"
    echo ""
else
    echo "=========================================="
    echo "ERROR: Some packages failed to import"
    echo "=========================================="
    echo ""
    echo "Please review the error messages above."
    exit 1
fi
