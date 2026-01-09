#!/bin/bash

# Quick setup script for atlas_kd conda environment

echo "=========================================="
echo "ATLAS Top Tagging Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo "Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Found conda: $(which conda)"
echo ""

# Ask user for Python version
echo "Which Python version would you like to use?"
echo "  1) Python 3.10 (recommended)"
echo "  2) Python 3.9"
read -p "Enter choice [1-2]: " python_choice

case $python_choice in
    1)
        PYTHON_VERSION="3.10"
        ;;
    2)
        PYTHON_VERSION="3.9"
        ;;
    *)
        echo "Invalid choice. Using Python 3.10 (default)"
        PYTHON_VERSION="3.10"
        ;;
esac

echo ""
echo "Creating conda environment: atlas_kd (Python $PYTHON_VERSION)"
conda create -n atlas_kd python=$PYTHON_VERSION -y

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment"
    exit 1
fi

echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate atlas_kd

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate environment"
    exit 1
fi

echo ""
echo "Which PyTorch installation would you like?"
echo "  1) GPU with CUDA 11.8"
echo "  2) GPU with CUDA 12.1"
echo "  3) CPU only"
read -p "Enter choice [1-3]: " pytorch_choice

case $pytorch_choice in
    1)
        echo "Installing PyTorch with CUDA 11.8..."
        conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1..."
        conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
        ;;
    3)
        echo "Installing PyTorch (CPU only)..."
        conda install pytorch torchvision cpuonly -c pytorch -y
        ;;
    *)
        echo "Invalid choice. Installing CPU version (default)"
        conda install pytorch torchvision cpuonly -c pytorch -y
        ;;
esac

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi

echo ""
echo "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')" || exit 1
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" || true
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" || true
fi

python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || exit 1
python -c "import h5py; print(f'h5py: {h5py.__version__}')" || exit 1
python -c "import energyflow; print(f'energyflow: {energyflow.__version__}')" || exit 1
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" || exit 1
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')" || exit 1

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment 'atlas_kd' has been created and configured."
echo ""
echo "To use it:"
echo "  conda activate atlas_kd"
echo ""
echo "To test with a single transformer run:"
echo "  python transformer_runner.py --run_name test --device cpu"
echo ""
echo "To submit SLURM jobs:"
echo "  ./submit_all_hyperparameter_jobs.sh"
echo ""
echo "See ENVIRONMENT_SETUP.md for more details."
echo ""
