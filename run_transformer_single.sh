#!/bin/bash

#SBATCH --job-name=transformer_kd
#SBATCH --output=transformer_logs/transformer_%j.out
#SBATCH --error=transformer_logs/transformer_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier3

# Create log directory
mkdir -p transformer_logs

# Print job info
echo "=========================================="
echo "Transformer KD Training - ATLAS Top Tagging"
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
TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
RUN_NAME=${RUN_NAME:-"default"}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  Temperature Init: $TEMP_INIT"
if [ -n "$TEMP_FINAL" ]; then
    echo "  Temperature Final: $TEMP_FINAL (annealing)"
else
    echo "  Temperature Final: constant"
fi
echo "  Alpha Init: $ALPHA_INIT"
if [ -n "$ALPHA_FINAL" ]; then
    echo "  Alpha Final: $ALPHA_FINAL (scheduling)"
else
    echo "  Alpha Final: constant"
fi
echo ""

# Build command with optional parameters
CMD="python transformer_runner.py \
    --save_dir checkpoints/transformer_search \
    --run_name $RUN_NAME \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT"

# Add optional temperature final
if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi

# Add optional alpha final
if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
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
    echo "Results saved to: checkpoints/transformer_search/$RUN_NAME"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
