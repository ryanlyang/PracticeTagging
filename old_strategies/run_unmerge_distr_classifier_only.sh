#!/bin/bash

#SBATCH --job-name=unmerge_distr_cls
#SBATCH --output=unmerge_distr_kfold_logs/unmerge_distr_cls_%j.out
#SBATCH --error=unmerge_distr_kfold_logs/unmerge_distr_cls_%j.err
#SBATCH --time=2-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_distr_kfold_logs

echo "=========================================="
echo "Unmerge Distr Classifier-Only"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

source ~/.bashrc
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold"}
RUN_NAME=${RUN_NAME:-"kfold_run"}
CACHE_PATH=${CACHE_PATH:-"$SAVE_DIR/$RUN_NAME/unmerged_cache.npz"}
MC_CACHE_PATH=${MC_CACHE_PATH:-"$SAVE_DIR/$RUN_NAME/unmerged_mc_cache.npz"}
TEACHER_CKPT=${TEACHER_CKPT:-"$SAVE_DIR/$RUN_NAME/teacher.pt"}
BASELINE_CKPT=${BASELINE_CKPT:-"$SAVE_DIR/$RUN_NAME/baseline.pt"}
SKIP_BASELINE=${SKIP_BASELINE:-0}
MC_SAMPLES=${MC_SAMPLES:-1}
MC_CONS_W=${MC_CONS_W:-0.1}
NO_SELF_TRAIN=${NO_SELF_TRAIN:-0}
ALPHA_KD=${ALPHA_KD:-""}
KD_TEMP=${KD_TEMP:-""}
ALPHA_ATTN=${ALPHA_ATTN:-""}
ALPHA_REP=${ALPHA_REP:-""}
ALPHA_NCE=${ALPHA_NCE:-""}
TAU_NCE=${TAU_NCE:-""}
NO_CONF_KD=${NO_CONF_KD:-0}
NO_ADAPT_ALPHA=${NO_ADAPT_ALPHA:-0}
ALPHA_WARMUP=${ALPHA_WARMUP:-""}
ALPHA_STABLE_PAT=${ALPHA_STABLE_PAT:-""}
ALPHA_STABLE_DELTA=${ALPHA_STABLE_DELTA:-""}
ALPHA_WARMUP_MIN_EPOCHS=${ALPHA_WARMUP_MIN_EPOCHS:-""}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-0}

CMD="python unmerge_distr_model.py \
  --save_dir $SAVE_DIR \
  --run_name ${RUN_NAME}_cls_${TAG:-base} \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --classifier_only \
  --load_unmerged_cache $CACHE_PATH \
  --mc_samples $MC_SAMPLES \
  --mc_consistency_weight $MC_CONS_W \
  --teacher_checkpoint $TEACHER_CKPT"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ -f "$BASELINE_CKPT" ] && [ "$SKIP_BASELINE" -eq 0 ]; then
  CMD="$CMD --baseline_checkpoint $BASELINE_CKPT"
else
  CMD="$CMD --skip_baseline"
fi

if [ "$NO_SELF_TRAIN" -eq 1 ]; then
  CMD="$CMD --no_self_train"
fi

if [ -n "$ALPHA_KD" ]; then
  CMD="$CMD --alpha_kd $ALPHA_KD"
fi

if [ -n "$KD_TEMP" ]; then
  CMD="$CMD --kd_temp $KD_TEMP"
fi

if [ -n "$ALPHA_ATTN" ]; then
  CMD="$CMD --alpha_attn $ALPHA_ATTN"
fi

if [ -n "$ALPHA_REP" ]; then
  CMD="$CMD --alpha_rep $ALPHA_REP"
fi

if [ -n "$ALPHA_NCE" ]; then
  CMD="$CMD --alpha_nce $ALPHA_NCE"
fi

if [ -n "$TAU_NCE" ]; then
  CMD="$CMD --tau_nce $TAU_NCE"
fi

if [ "$NO_CONF_KD" -eq 1 ]; then
  CMD="$CMD --no_conf_kd"
fi

if [ "$NO_ADAPT_ALPHA" -eq 1 ]; then
  CMD="$CMD --no_adaptive_alpha"
fi

if [ -n "$ALPHA_WARMUP" ]; then
  CMD="$CMD --alpha_warmup $ALPHA_WARMUP"
fi

if [ -n "$ALPHA_STABLE_PAT" ]; then
  CMD="$CMD --alpha_stable_patience $ALPHA_STABLE_PAT"
fi

if [ -n "$ALPHA_STABLE_DELTA" ]; then
  CMD="$CMD --alpha_stable_delta $ALPHA_STABLE_DELTA"
fi

if [ -n "$ALPHA_WARMUP_MIN_EPOCHS" ]; then
  CMD="$CMD --alpha_warmup_min_epochs $ALPHA_WARMUP_MIN_EPOCHS"
fi

if [ "$NO_DISTRIBUTIONAL" -eq 1 ]; then
  CMD="$CMD --no_distributional"
fi

if [ "$MC_SAMPLES" -gt 1 ]; then
  CMD="$CMD --load_mc_cache $MC_CACHE_PATH"
fi

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CMD="$CMD --device cuda"
else
  CMD="$CMD --device cpu"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Classifier-only run completed"
else
  echo "Classifier-only run failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
