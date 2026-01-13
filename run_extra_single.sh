#!/bin/bash

#SBATCH --job-name=extra_cons
#SBATCH --output=extra_logs/job_%j.out
#SBATCH --error=extra_logs/job_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# Single hyperparameter run for extra_first_teach.py
# Uses pre-trained Teacher, Baseline, Union models
# Only trains the Consistency model with specified hyperparameters

mkdir -p extra_logs

echo "=========================================="
echo "Extra First Teach - Single Hyperparameter Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Run Name: $RUN_NAME"
echo "Start Time: $(date)"
echo ""
echo "Hyperparameters:"
echo "  rampup_frac:      $RAMPUP_FRAC"
echo "  lambda_prob:      $LAMBDA_PROB"
echo "  lambda_emb:       $LAMBDA_EMB"
echo "  attention_epoch:  $ATTENTION_EPOCH"
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
echo "Running extra_first_teach.py..."
echo ""

# Build command to run extra_first_teach.py
# Use pre-trained teacher checkpoint to skip teacher training
# Teacher, Baseline, Union will load from shared_models
# Only Consistency will be trained with these hyperparameters
CMD="python extra_first_teach.py \
    --save_dir checkpoints/extra_search \
    --run_name $RUN_NAME \
    --hlt_seed1 123 \
    --hlt_seed2 456 \
    --lambda_prob $LAMBDA_PROB \
    --lambda_emb $LAMBDA_EMB \
    --rampup_frac $RAMPUP_FRAC \
    --conf_power 1.0 \
    --conf_min 0.0 \
    --attention_epoch $ATTENTION_EPOCH \
    --teacher_checkpoint checkpoints/extra_search/shared_models/teacher.pt \
    --skip_save_models \
    --device $DEVICE"

echo "Command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Run completed successfully!"
    echo "Results saved to: checkpoints/extra_search/$RUN_NAME/"
else
    echo "Run failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
