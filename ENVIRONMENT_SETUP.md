# Environment Setup for ATLAS Top Tagging

This guide will help you set up the `atlas_kd` conda environment for running the Knowledge Distillation hyperparameter search.

## Quick Setup

### 1. Create the conda environment

```bash
conda create -n atlas_kd python=3.10 -y
```

### 2. Activate the environment

```bash
conda activate atlas_kd
```

### 3. Install PyTorch (choose based on your system)

**For GPU (CUDA 11.8):**
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**For GPU (CUDA 12.1):**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**For CPU only:**
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import energyflow; import h5py; import sklearn; print('All packages imported successfully!')"
```

## Step-by-Step Installation

### Step 1: Create Environment

Create a new conda environment with Python 3.10:
```bash
conda create -n atlas_kd python=3.10 -y
```

### Step 2: Activate Environment

```bash
conda activate atlas_kd
```

You should see `(atlas_kd)` in your terminal prompt.

### Step 3: Install PyTorch

Visit https://pytorch.org/get-started/locally/ to get the exact command for your system.

**Example for common setups:**

- **SLURM cluster with CUDA 11.8:**
  ```bash
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
  ```

- **SLURM cluster with CUDA 12.1:**
  ```bash
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
  ```

- **Local machine / CPU only:**
  ```bash
  conda install pytorch torchvision cpuonly -c pytorch -y
  ```

### Step 4: Install Other Dependencies

Install the remaining packages from requirements.txt:
```bash
pip install -r requirements.txt
```

This will install:
- h5py (HDF5 file handling)
- numpy, pandas (data processing)
- tensorflow (for EFN models)
- energyflow (physics ML library)
- scikit-learn, scipy (ML utilities)
- matplotlib, seaborn (plotting)
- tqdm (progress bars)

### Step 5: Verify Installation

Test that everything is installed correctly:

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

# Check TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Check other packages
python -c "import h5py; print(f'h5py version: {h5py.__version__}')"
python -c "import energyflow; print(f'energyflow version: {energyflow.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')"

echo "All packages installed successfully!"
```

## Testing on SLURM

Once the environment is set up, test it with a single job:

```bash
# Submit a test job
sbatch --export=ALL,TEMP_INIT=7.0,ALPHA_INIT=0.5,RUN_NAME="test_run" run_transformer_single.sh

# Check job status
squeue -u $USER

# Watch the output
tail -f transformer_logs/transformer_*.out
```

## Running the Full Hyperparameter Search

Once you've verified the environment works:

```bash
./submit_all_hyperparameter_jobs.sh
```

This will submit ~144 jobs to the SLURM queue.

## Troubleshooting

### Problem: `conda: command not found`

**Solution:** Initialize conda in your shell:
```bash
source /path/to/conda/etc/profile.d/conda.sh
```

Or add to your `~/.bashrc`:
```bash
# >>> conda initialize >>>
# Add conda initialization here
# <<< conda initialize <<<
```

### Problem: PyTorch can't find CUDA

**Solution:** Check your CUDA version and reinstall PyTorch with matching CUDA version:
```bash
# Check CUDA version
nvcc --version
# or
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda install pytorch torchvision pytorch-cuda=XX.X -c pytorch -c nvidia -y
```

### Problem: TensorFlow GPU not working

**Solution:** TensorFlow GPU support can be tricky. If you only need transformers (PyTorch), you can use CPU-only TensorFlow:
```bash
pip install tensorflow-cpu
```

### Problem: Import errors for energyflow

**Solution:** Make sure you're using Python 3.10 or earlier (energyflow may not support 3.11+):
```bash
python --version  # Should show 3.10.x
```

### Problem: SLURM jobs fail immediately

**Solution:** Check the error log:
```bash
cat transformer_logs/transformer_*.err
```

Common issues:
- Environment not activated (check `conda activate atlas_kd` in the script)
- Missing data files (check paths in transformer_runner.py)
- Memory/time limits too low (adjust in run_transformer_single.sh)

## Environment Management

### Deactivate environment
```bash
conda deactivate
```

### Remove environment (if needed)
```bash
conda env remove -n atlas_kd
```

### Export environment (for reproducibility)
```bash
conda env export > environment.yml
```

### Recreate from exported environment
```bash
conda env create -f environment.yml
```

## Notes

- The SLURM script (`run_transformer_single.sh`) automatically activates the `atlas_kd` environment
- GPU will be auto-detected and used if available
- Each job saves its own results to avoid conflicts
- All results are logged to `checkpoints/transformer_search/hyperparameter_search_results.txt`
