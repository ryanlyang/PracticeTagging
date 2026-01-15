#!/bin/bash

#SBATCH --job-name=more_const_teach
#SBATCH --output=transformer_logs/more_const_%j.out
#SBATCH --error=transformer_logs/more_const_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# Create log directory
mkdir -p transformer_logs

# Print job info
echo "=========================================="
echo "Multi-View Consistency + KD - ATLAS Top Tagging"
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
N_HLT_VIEWS=${N_HLT_VIEWS:-2}
LAMBDA_PROB=${LAMBDA_PROB:-1.0}
LAMBDA_EMB=${LAMBDA_EMB:-0.25}
RAMPUP_FRAC=${RAMPUP_FRAC:-0.2}
CONF_POWER=${CONF_POWER:-1.0}
CONF_MIN=${CONF_MIN:-0.0}
TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
NO_CONF_KD=${NO_CONF_KD:-0}
HLT_SEEDS=${HLT_SEEDS:-""}
HLT_SEED_BASE=${HLT_SEED_BASE:-123}
HLT_SEED_STEP=${HLT_SEED_STEP:-333}
RUN_NAME=${RUN_NAME:-"default"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/transformer_twohlt"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-""}
TEACHER_CKPT=${TEACHER_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  n_hlt_views: $N_HLT_VIEWS"
echo "  lambda_prob: $LAMBDA_PROB"
echo "  lambda_emb: $LAMBDA_EMB"
echo "  rampup_frac: $RAMPUP_FRAC"
echo "  conf_power: $CONF_POWER"
echo "  conf_min: $CONF_MIN"
echo "  temp_init: $TEMP_INIT"
if [ -n "$TEMP_FINAL" ]; then
    echo "  temp_final: $TEMP_FINAL (annealing)"
else
    echo "  temp_final: constant"
fi
echo "  alpha_init: $ALPHA_INIT"
if [ -n "$ALPHA_FINAL" ]; then
    echo "  alpha_final: $ALPHA_FINAL (scheduling)"
else
    echo "  alpha_final: constant"
fi
echo "  alpha_attn: $ALPHA_ATTN"
echo "  alpha_rep: $ALPHA_REP"
echo "  alpha_nce: $ALPHA_NCE"
echo "  tau_nce: $TAU_NCE"
echo "  conf_weighted_kd: $([ "$NO_CONF_KD" -eq 1 ] && echo no || echo yes)"
echo ""

CMD="python more_const_teach.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --n_hlt_views $N_HLT_VIEWS \
    --lambda_prob $LAMBDA_PROB \
    --lambda_emb $LAMBDA_EMB \
    --rampup_frac $RAMPUP_FRAC \
    --conf_power $CONF_POWER \
    --conf_min $CONF_MIN \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT \
    --alpha_attn $ALPHA_ATTN \
    --alpha_rep $ALPHA_REP \
    --alpha_nce $ALPHA_NCE \
    --tau_nce $TAU_NCE \
    --hlt_seed_base $HLT_SEED_BASE \
    --hlt_seed_step $HLT_SEED_STEP"

if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi

if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
fi

if [ -n "$HLT_SEEDS" ]; then
    CMD="$CMD --hlt_seeds $HLT_SEEDS"
fi

if [ -n "$TRAIN_PATH" ]; then
    CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ -n "$N_TRAIN_JETS" ]; then
    CMD="$CMD --n_train_jets $N_TRAIN_JETS"
fi

if [ -n "$MAX_CONSTITS" ]; then
    CMD="$CMD --max_constits $MAX_CONSTITS"
fi

if [ -n "$TEACHER_CKPT" ]; then
    CMD="$CMD --teacher_checkpoint $TEACHER_CKPT"
fi

if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
    CMD="$CMD --skip_save_models"
fi

if [ "$NO_CONF_KD" -eq 1 ]; then
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
