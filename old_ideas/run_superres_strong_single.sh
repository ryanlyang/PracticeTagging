#!/bin/bash

#SBATCH --job-name=superres_strong
#SBATCH --output=superres_strong_logs/superres_strong_%j.out
#SBATCH --error=superres_strong_logs/superres_strong_%j.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p superres_strong_logs

echo "=========================================="
echo "SuperRes Strong (HLT -> SR -> Offline Classifier)"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/superres_strong_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
SR_MODE=${SR_MODE:-"both"}

SR_SAMPLES=${SR_SAMPLES:-4}
SR_EVAL_SAMPLES=${SR_EVAL_SAMPLES:-4}
SR_EPOCHS=${SR_EPOCHS:-100}
SR_LR=${SR_LR:-5e-4}
SR_MASK_THRESHOLD=${SR_MASK_THRESHOLD:-0.5}

GEN_EMBED_DIM=${GEN_EMBED_DIM:-128}
GEN_NUM_HEADS=${GEN_NUM_HEADS:-8}
GEN_NUM_LAYERS=${GEN_NUM_LAYERS:-4}
GEN_DEC_LAYERS=${GEN_DEC_LAYERS:-2}
GEN_FF_DIM=${GEN_FF_DIM:-512}
GEN_DROPOUT=${GEN_DROPOUT:-0.1}
GEN_LATENT_DIM=${GEN_LATENT_DIM:-32}
GEN_MIX_COMPONENTS=${GEN_MIX_COMPONENTS:-8}
GEN_MIN_SIGMA=${GEN_MIN_SIGMA:-1e-3}
GEN_LR=${GEN_LR:-3e-4}
GEN_EPOCHS=${GEN_EPOCHS:-60}
GEN_PATIENCE=${GEN_PATIENCE:-20}
GEN_LAMBDA_FEAT=${GEN_LAMBDA_FEAT:-1.0}
GEN_LAMBDA_MASK=${GEN_LAMBDA_MASK:-0.5}
GEN_LAMBDA_MULT=${GEN_LAMBDA_MULT:-0.1}
GEN_LAMBDA_STATS=${GEN_LAMBDA_STATS:-0.1}
GEN_LAMBDA_PERC=${GEN_LAMBDA_PERC:-0.2}
GEN_LAMBDA_LOGIT=${GEN_LAMBDA_LOGIT:-0.1}

LOSS_SUP=${LOSS_SUP:-1.0}
LOSS_CONS_PROB=${LOSS_CONS_PROB:-0.1}
LOSS_CONS_EMB=${LOSS_CONS_EMB:-0.05}

# Locked KD/consistency hyperparameters
CONF_POWER=${CONF_POWER:-2.0}
CONF_MIN=${CONF_MIN:-0.0}
TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD:-1}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  sr_mode: $SR_MODE"
echo "  sr_samples (train/eval): $SR_SAMPLES / $SR_EVAL_SAMPLES"
echo "  mix_components/min_sigma: $GEN_MIX_COMPONENTS / $GEN_MIN_SIGMA"
echo "  gen_layers/dec_layers: $GEN_NUM_LAYERS / $GEN_DEC_LAYERS"
echo "  gen_lambda_perc/logit: $GEN_LAMBDA_PERC / $GEN_LAMBDA_LOGIT"
echo "  KD temp/alpha: $TEMP_INIT / $ALPHA_INIT"
echo ""

CMD="python SuperResStrong.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --sr_mode $SR_MODE \
    --sr_samples $SR_SAMPLES \
    --sr_eval_samples $SR_EVAL_SAMPLES \
    --sr_epochs $SR_EPOCHS \
    --sr_lr $SR_LR \
    --sr_mask_threshold $SR_MASK_THRESHOLD \
    --gen_embed_dim $GEN_EMBED_DIM \
    --gen_num_heads $GEN_NUM_HEADS \
    --gen_num_layers $GEN_NUM_LAYERS \
    --gen_dec_layers $GEN_DEC_LAYERS \
    --gen_ff_dim $GEN_FF_DIM \
    --gen_dropout $GEN_DROPOUT \
    --gen_latent_dim $GEN_LATENT_DIM \
    --gen_mix_components $GEN_MIX_COMPONENTS \
    --gen_min_sigma $GEN_MIN_SIGMA \
    --gen_lr $GEN_LR \
    --gen_epochs $GEN_EPOCHS \
    --gen_patience $GEN_PATIENCE \
    --gen_lambda_feat $GEN_LAMBDA_FEAT \
    --gen_lambda_mask $GEN_LAMBDA_MASK \
    --gen_lambda_mult $GEN_LAMBDA_MULT \
    --gen_lambda_stats $GEN_LAMBDA_STATS \
    --gen_lambda_perc $GEN_LAMBDA_PERC \
    --gen_lambda_logit $GEN_LAMBDA_LOGIT \
    --loss_sup $LOSS_SUP \
    --loss_cons_prob $LOSS_CONS_PROB \
    --loss_cons_emb $LOSS_CONS_EMB \
    --conf_power $CONF_POWER \
    --conf_min $CONF_MIN \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT \
    --alpha_attn $ALPHA_ATTN \
    --alpha_rep $ALPHA_REP \
    --alpha_nce $ALPHA_NCE \
    --tau_nce $TAU_NCE"

if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi
if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
fi
if [ "$CONF_WEIGHTED_KD" -eq 0 ]; then
    CMD="$CMD --no_conf_kd"
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
