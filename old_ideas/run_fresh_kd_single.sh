#!/bin/bash

#SBATCH --job-name=fresh_kd
#SBATCH --output=fresh_kd_logs/fresh_kd_%j.out
#SBATCH --error=fresh_kd_logs/fresh_kd_%j.err
#SBATCH --time=3-16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p fresh_kd_logs

echo "=========================================="
echo "Fresh KD Training - ATLAS Top Tagging"
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

RUN_NAME=${RUN_NAME:-"default"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/fresh_kd_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-1}

TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
TEACHER_ENSEMBLE_CKPTS=${TEACHER_ENSEMBLE_CKPTS:-""}
USE_ENSEMBLE=${USE_ENSEMBLE:-0}

TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}

ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
ALPHA_REL=${ALPHA_REL:-0.0}
TAU_NCE=${TAU_NCE:-0.10}
NO_CONF_KD=${NO_CONF_KD:-0}

CALIBRATE_TEACHER=${CALIBRATE_TEACHER:-0}
TEACHER_CALIB_MAX_ITER=${TEACHER_CALIB_MAX_ITER:-50}
EMA_TEACHER=${EMA_TEACHER:-0}
EMA_DECAY=${EMA_DECAY:-0.999}

ADAPTIVE_ALPHA=${ADAPTIVE_ALPHA:-0}
ALPHA_WARMUP=${ALPHA_WARMUP:-0.0}
ALPHA_STABLE_PATIENCE=${ALPHA_STABLE_PATIENCE:-3}
ALPHA_STABLE_DELTA=${ALPHA_STABLE_DELTA:-1e-4}
ALPHA_WARMUP_MIN_EPOCHS=${ALPHA_WARMUP_MIN_EPOCHS:-3}

SELF_TRAIN=${SELF_TRAIN:-0}
SELF_TRAIN_SOURCE=${SELF_TRAIN_SOURCE:-"teacher"}
SELF_TRAIN_EPOCHS=${SELF_TRAIN_EPOCHS:-5}
SELF_TRAIN_LR=${SELF_TRAIN_LR:-1e-4}
SELF_TRAIN_CONF_MIN=${SELF_TRAIN_CONF_MIN:-0.0}
SELF_TRAIN_CONF_POWER=${SELF_TRAIN_CONF_POWER:-1.0}
SELF_TRAIN_HARD=${SELF_TRAIN_HARD:-0}
SELF_TRAIN_PATIENCE=${SELF_TRAIN_PATIENCE:-5}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  temp_init: $TEMP_INIT"
if [ -n "$TEMP_FINAL" ]; then
    echo "  temp_final: $TEMP_FINAL"
else
    echo "  temp_final: constant"
fi
echo "  alpha_init: $ALPHA_INIT"
if [ -n "$ALPHA_FINAL" ]; then
    echo "  alpha_final: $ALPHA_FINAL"
else
    echo "  alpha_final: constant"
fi
echo "  alpha_attn/rep/nce/rel: $ALPHA_ATTN / $ALPHA_REP / $ALPHA_NCE / $ALPHA_REL"
echo "  adaptive_alpha: $ADAPTIVE_ALPHA"
echo "  calibrate_teacher: $CALIBRATE_TEACHER"
echo "  ema_teacher: $EMA_TEACHER (decay=$EMA_DECAY)"
echo "  self_train: $SELF_TRAIN (source=$SELF_TRAIN_SOURCE)"
echo ""

CMD="python fresh_KD.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT \
    --alpha_attn $ALPHA_ATTN \
    --alpha_rep $ALPHA_REP \
    --alpha_nce $ALPHA_NCE \
    --alpha_rel $ALPHA_REL \
    --tau_nce $TAU_NCE"

if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi
if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
fi
if [ "$NO_CONF_KD" -eq 1 ]; then
    CMD="$CMD --no_conf_kd"
fi
if [ "$CALIBRATE_TEACHER" -eq 1 ]; then
    CMD="$CMD --calibrate_teacher --teacher_calib_max_iter $TEACHER_CALIB_MAX_ITER"
fi
if [ "$EMA_TEACHER" -eq 1 ]; then
    CMD="$CMD --ema_teacher --ema_decay $EMA_DECAY"
fi
if [ "$ADAPTIVE_ALPHA" -eq 1 ]; then
    CMD="$CMD --adaptive_alpha --alpha_warmup $ALPHA_WARMUP --alpha_stable_patience $ALPHA_STABLE_PATIENCE --alpha_stable_delta $ALPHA_STABLE_DELTA --alpha_warmup_min_epochs $ALPHA_WARMUP_MIN_EPOCHS"
fi
if [ "$SELF_TRAIN" -eq 1 ]; then
    CMD="$CMD --self_train --self_train_source $SELF_TRAIN_SOURCE --self_train_epochs $SELF_TRAIN_EPOCHS --self_train_lr $SELF_TRAIN_LR --self_train_conf_min $SELF_TRAIN_CONF_MIN --self_train_conf_power $SELF_TRAIN_CONF_POWER --self_train_patience $SELF_TRAIN_PATIENCE"
    if [ "$SELF_TRAIN_HARD" -eq 1 ]; then
        CMD="$CMD --self_train_hard"
    fi
fi
if [ "$USE_ENSEMBLE" -eq 1 ] && [ -n "$TEACHER_ENSEMBLE_CKPTS" ]; then
    CMD="$CMD --teacher_ensemble_checkpoints $TEACHER_ENSEMBLE_CKPTS"
elif [ -n "$TEACHER_CKPT" ]; then
    CMD="$CMD --teacher_checkpoint $TEACHER_CKPT"
fi
if [ -n "$BASELINE_CKPT" ]; then
    CMD="$CMD --baseline_checkpoint $BASELINE_CKPT"
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
if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
    CMD="$CMD --skip_save_models"
fi

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
