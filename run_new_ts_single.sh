#!/bin/bash

#SBATCH --job-name=new_ts_kd
#SBATCH --output=new_ts_logs/job_%j.out
#SBATCH --error=new_ts_logs/job_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# Create log directory
mkdir -p new_ts_logs

# Print job info
echo "=========================================="
echo "New Teacher-Student KD Training"
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

# Hyperparameter values (passed from submission script)
RUN_NAME=${RUN_NAME:-"default"}
TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
NO_CONF_KD=${NO_CONF_KD:-""}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  Temperature: $TEMP_INIT → ${TEMP_FINAL:-constant}"
echo "  Alpha KD: $ALPHA_INIT → ${ALPHA_FINAL:-constant}"
echo "  Alpha Attn: $ALPHA_ATTN"
echo "  Alpha Rep: $ALPHA_REP"
echo "  Alpha NCE: $ALPHA_NCE"
echo "  Tau NCE: $TAU_NCE"
echo "  Conf-weighted KD: ${NO_CONF_KD:-ON}"
echo ""

# Build command
CMD="python new_teacher_student.py \
    --save_dir checkpoints/new_ts_search \
    --run_name $RUN_NAME \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT \
    --alpha_attn $ALPHA_ATTN \
    --alpha_rep $ALPHA_REP \
    --alpha_nce $ALPHA_NCE \
    --tau_nce $TAU_NCE \
    --teacher_checkpoint checkpoints/new_ts_search/shared_models/teacher.pt \
    --baseline_checkpoint checkpoints/new_ts_search/shared_models/baseline.pt \
    --skip_save_models"

# Add optional temperature final
if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi

# Add optional alpha final
if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
fi

# Add no_conf_kd flag if specified
if [ -n "$NO_CONF_KD" ]; then
    CMD="$CMD --no_conf_kd"
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

# Run training
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    echo "Results saved to: checkpoints/new_ts_search/$RUN_NAME"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
