#!/bin/bash

# Quick fix for PyTorch Intel MKL issue

echo "Fixing PyTorch environment..."

# Activate environment
source ~/.bashrc
conda activate atlas_kd

# The issue is mixing conda and pip for torch
# Solution: Remove everything and reinstall via conda only

echo "Step 1: Removing conflicting packages..."
conda remove -y --force pytorch torchvision numpy scipy mkl mkl-service intel-openmp

echo "Step 2: Reinstalling PyTorch and dependencies via conda..."
conda install -y -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8

echo "Step 3: Installing numpy 1.26 (compatible with torch)..."
conda install -y "numpy<2.0"

echo "Step 4: Installing other scientific packages via conda..."
conda install -y scipy scikit-learn matplotlib seaborn

echo "Step 5: Installing remaining packages via pip..."
pip install h5py pandas energyflow tqdm tensorflow

echo ""
echo "Testing imports..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)"
python -c "import h5py; print('✓ h5py:', h5py.__version__)"
python -c "import energyflow; print('✓ energyflow:', energyflow.__version__)"

echo ""
echo "Environment fixed! You can now run: sbatch train_shared_models.sh"
