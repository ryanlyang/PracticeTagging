#!/bin/bash

#SBATCH --job-name=hlt_generation
#SBATCH --output=hlt_gen_logs/hlt_gen_%j.out
#SBATCH --error=hlt_gen_logs/hlt_gen_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# Create log directory
mkdir -p hlt_gen_logs

# Print job info
echo "=========================================="
echo "HLT Generation + KD + Consistency"
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
TEMP_INIT=${TEMP_INIT:-5.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_CONS_PROB=${ALPHA_CONS_PROB:-0.5}
ALPHA_CONS_EMB=${ALPHA_CONS_EMB:-0.5}
CONS_CONF_POWER=${CONS_CONF_POWER:-2.0}
CONS_CONF_MIN=${CONS_CONF_MIN:-0.0}
CONS_RAMPUP_FRAC=${CONS_RAMPUP_FRAC:-0.2}
GEN_VIEWS=${GEN_VIEWS:-3}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
GEN_LAMBDA_MASK=${GEN_LAMBDA_MASK:-1.0}
GEN_LAMBDA_STATS=${GEN_LAMBDA_STATS:-0.1}
GEN_MIN_SIGMA=${GEN_MIN_SIGMA:-0.02}
GEN_EPOCHS=${GEN_EPOCHS:-30}
GEN_LR=${GEN_LR:-1e-3}
GEN_GLOBAL_NOISE=${GEN_GLOBAL_NOISE:-0.05}
RUN_NAME=${RUN_NAME:-"default"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/hlt_generation_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
GENERATOR_CKPT=${GENERATOR_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
NO_CONF_KD=${NO_CONF_KD:-0}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
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
echo "  cons_prob: $ALPHA_CONS_PROB"
echo "  cons_emb: $ALPHA_CONS_EMB"
echo "  cons_conf_power: $CONS_CONF_POWER"
echo "  alpha_attn: $ALPHA_ATTN"
echo "  alpha_rep: $ALPHA_REP"
echo "  alpha_nce: $ALPHA_NCE"
echo "  tau_nce: $TAU_NCE"
echo "  gen_views: $GEN_VIEWS"
echo "  gen_min_sigma: $GEN_MIN_SIGMA"
echo "  gen_lambda_mask: $GEN_LAMBDA_MASK"
echo "  gen_lambda_stats: $GEN_LAMBDA_STATS"
echo ""

CMD="python HLT_generation.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT \
    --alpha_attn $ALPHA_ATTN \
    --alpha_rep $ALPHA_REP \
    --alpha_nce $ALPHA_NCE \
    --tau_nce $TAU_NCE \
    --alpha_cons_prob $ALPHA_CONS_PROB \
    --alpha_cons_emb $ALPHA_CONS_EMB \
    --cons_conf_power $CONS_CONF_POWER \
    --cons_conf_min $CONS_CONF_MIN \
    --cons_rampup_frac $CONS_RAMPUP_FRAC \
    --gen_views $GEN_VIEWS \
    --gen_min_sigma $GEN_MIN_SIGMA \
    --gen_lambda_mask $GEN_LAMBDA_MASK \
    --gen_lambda_stats $GEN_LAMBDA_STATS \
    --gen_epochs $GEN_EPOCHS \
    --gen_lr $GEN_LR \
    --gen_global_noise $GEN_GLOBAL_NOISE"

if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi

if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
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

if [ -n "$BASELINE_CKPT" ]; then
    CMD="$CMD --baseline_checkpoint $BASELINE_CKPT"
fi

if [ -n "$GENERATOR_CKPT" ]; then
    CMD="$CMD --generator_checkpoint $GENERATOR_CKPT"
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
