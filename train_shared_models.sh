#!/bin/bash

#SBATCH --job-name=train_shared
#SBATCH --output=transformer_logs/train_shared_%j.out
#SBATCH --error=transformer_logs/train_shared_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier3

# This script trains the teacher and baseline models ONCE
# These will be reused for all hyperparameter search runs

mkdir -p transformer_logs
mkdir -p checkpoints/transformer_search/shared_models

echo "=========================================="
echo "Training Shared Teacher and Baseline Models"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate atlas_kd

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print environment
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Check if GPU is available
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "Using GPU for training"
else
    DEVICE="cpu"
    echo "Using CPU for training"
fi

echo ""
echo "Training teacher and baseline models..."
echo "These will be saved to: checkpoints/transformer_search/shared_models/"
echo "All hyperparameter search runs will reuse these models."
echo ""

# Run training (this will train teacher, baseline, and one student with default hyperparameters)
python transformer_runner.py \
    --save_dir checkpoints/transformer_search \
    --run_name shared_models \
    --temp_init 7.0 \
    --alpha_init 0.5 \
    --device $DEVICE

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Shared models saved to:"
    echo "  Teacher:  checkpoints/transformer_search/shared_models/teacher.pt"
    echo "  Baseline: checkpoints/transformer_search/shared_models/baseline.pt"
    echo ""
    echo "To run hyperparameter search using these models:"
    echo "  ./submit_all_hyperparameter_jobs.sh"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
