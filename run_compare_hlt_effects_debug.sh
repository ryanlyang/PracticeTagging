#!/bin/bash

#SBATCH --job-name=compare_hlt
#SBATCH --output=compare_hlt_logs/compare_hlt_%j.out
#SBATCH --error=compare_hlt_logs/compare_hlt_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=debug
#SBATCH --gres=gpu:1

# Create log directory
mkdir -p compare_hlt_logs

echo "=========================================="
echo "Compare HLT Effects - ATLAS Top Tagging"
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
cd "$SLURM_SUBMIT_DIR"

# Print environment
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Hyperparameter values (optional overrides from env)
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SAVE_DIR=${SAVE_DIR:-"checkpoints/compare_hlt_effects"}
RUN_NAME=${RUN_NAME:-"compare_hlt_effects"}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
PATIENCE=${PATIENCE:-15}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  n_train_jets: $N_TRAIN_JETS"
echo "  max_constits: $MAX_CONSTITS"
echo "  epochs: $EPOCHS"
echo "  batch_size: $BATCH_SIZE"
echo "  lr: $LR"
echo "  weight_decay: $WEIGHT_DECAY"
echo "  warmup_epochs: $WARMUP_EPOCHS"
echo "  patience: $PATIENCE"
echo ""

CMD="python compare_hlt_effects.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --n_train_jets $N_TRAIN_JETS \
    --max_constits $MAX_CONSTITS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --patience $PATIENCE"

if [ -n "$TRAIN_PATH" ]; then
    CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
    CMD="$CMD --skip_save_models"
fi

# Check if GPU is available and add device flag
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CMD="$CMD --device cuda"
    echo "Using GPU for training"
else
    CMD="$CMD --device cpu"
    echo "Using CPU for training"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    echo "Results saved to: $SAVE_DIR/$RUN_NAME"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
