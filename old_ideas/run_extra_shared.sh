#!/bin/bash

#SBATCH --job-name=extra_shared
#SBATCH --output=extra_logs/shared_%j.out
#SBATCH --error=extra_logs/shared_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# This script trains the Teacher, Baseline, and Union models ONCE for extra_first_teach.py
# These will be reused for all hyperparameter search runs

mkdir -p extra_logs
mkdir -p checkpoints/extra_search/shared_models

echo "=========================================="
echo "Training Shared Models for Extra First Teach"
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
echo "Training Teacher, Baseline, and Union models..."
echo "These will be saved to: checkpoints/extra_search/shared_models/"
echo "All hyperparameter search runs will reuse these models."
echo ""

# Run training with default hyperparameters (just to create the shared models)
# We'll use --skip_save_models to avoid saving the consistency model here
python extra_first_teach.py \
    --save_dir checkpoints/extra_search \
    --run_name shared_models \
    --hlt_seed1 123 \
    --hlt_seed2 456 \
    --lambda_prob 1.0 \
    --lambda_emb 0.25 \
    --rampup_frac 0.2 \
    --conf_power 1.0 \
    --conf_min 0.0 \
    --attention_epoch 0 \
    --device $DEVICE

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Shared models saved to:"
    echo "  Teacher:  checkpoints/extra_search/shared_models/teacher.pt"
    echo "  Baseline: checkpoints/extra_search/shared_models/baseline_hlt1.pt"
    echo "  Union:    checkpoints/extra_search/shared_models/union_hlt12.pt"
    echo ""
    echo "To run hyperparameter search using these models:"
    echo "  ./run_extra_grid_search.sh"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
