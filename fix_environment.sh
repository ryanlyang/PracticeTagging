#!/bin/bash

# Fix PyTorch + Intel MKL compatibility issue

echo "=========================================="
echo "Fixing PyTorch Environment"
echo "=========================================="
echo ""

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate atlas_kd

echo "Reinstalling PyTorch with compatible dependencies..."
echo ""

# Remove potentially conflicting packages
pip uninstall -y torch torchvision numpy sympy

# Reinstall with conda (more reliable dependency resolution)
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Reinstall numpy with compatible version
conda install -y "numpy<2.0"

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" && \
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" && \
python -c "import h5py; print(f'h5py: {h5py.__version__}')" && \
python -c "import energyflow; print(f'energyflow: {energyflow.__version__}')" && \
echo "" && \
echo "==========================================" && \
echo "Fix Complete!" && \
echo "=========================================="

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Some packages failed to import"
    echo ""
    echo "Try running:"
    echo "  conda activate atlas_kd"
    echo "  conda install -c conda-forge numpy=1.26"
    exit 1
fi
