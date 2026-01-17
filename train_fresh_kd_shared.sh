#!/bin/bash

#SBATCH --job-name=train_fresh_kd_shared
#SBATCH --output=fresh_kd_logs/train_shared_%j.out
#SBATCH --error=fresh_kd_logs/train_shared_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# Train shared teacher + baseline once for fresh_KD sweeps.

mkdir -p fresh_kd_logs
mkdir -p checkpoints/fresh_kd/shared_models

echo "=========================================="
echo "Training Shared Teacher and Baseline (fresh_KD)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

source ~/.bashrc
conda activate atlas_kd

cd $SLURM_SUBMIT_DIR

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

DEVICE="cpu"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "Using GPU for training"
else
    echo "Using CPU for training"
fi

TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}

CMD="python fresh_KD.py \
    --save_dir checkpoints/fresh_kd \
    --run_name shared_models \
    --temp_init 7.0 \
    --alpha_init 0.5 \
    --device $DEVICE"

if [ -n "$TRAIN_PATH" ]; then
    CMD="$CMD --train_path $TRAIN_PATH"
fi
if [ -n "$N_TRAIN_JETS" ]; then
    CMD="$CMD --n_train_jets $N_TRAIN_JETS"
fi
if [ -n "$MAX_CONSTITS" ]; then
    CMD="$CMD --max_constits $MAX_CONSTITS"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Shared models saved to:"
    echo "  Teacher:  checkpoints/fresh_kd/shared_models/teacher.pt"
    echo "  Baseline: checkpoints/fresh_kd/shared_models/baseline.pt"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
