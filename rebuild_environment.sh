#!/bin/bash

# Complete environment rebuild script for ATLAS Knowledge Distillation
# Fixes corrupted conda environment where packages are registered but not actually installed

echo "=========================================="
echo "Rebuilding ATLAS KD Environment"
echo "=========================================="
echo ""

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh

echo "Step 1: Removing corrupted environment..."
conda deactivate 2>/dev/null || true
conda env remove -n atlas_kd -y

echo ""
echo "Step 2: Creating fresh environment with Python 3.10..."
conda create -n atlas_kd python=3.10 -y

echo ""
echo "Step 3: Activating new environment..."
conda activate atlas_kd

echo ""
echo "Step 4: Installing PyTorch and core scientific packages via conda..."
# Install all conda packages in a single transaction for better dependency resolution
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
python -c "import torch; print('✓ PyTorch:', torch.__version__)" && \
python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available())" && \
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" && \
python -c "import scipy; print('✓ SciPy:', scipy.__version__)" && \
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" && \
python -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)" && \
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)" && \
python -c "import h5py; print('✓ h5py:', h5py.__version__)" && \
python -c "import pandas; print('✓ pandas:', pandas.__version__)" && \
python -c "import energyflow; print('✓ energyflow:', energyflow.__version__)" && \
echo "" && \
echo "=========================================="  && \
echo "Environment rebuild complete!" && \
echo "=========================================="  && \
echo "" && \
echo "You can now run:" && \
echo "  sbatch train_shared_models.sh"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Some packages failed to import"
    echo ""
    echo "Please check the error messages above and report the issue."
    exit 1
fi
