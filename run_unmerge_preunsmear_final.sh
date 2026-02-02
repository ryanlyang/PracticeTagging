#!/bin/bash

#SBATCH --job-name=unmerge_preunsmear
#SBATCH --output=unmerge_preunsmear_logs/unmerge_preunsmear_%j.out
#SBATCH --error=unmerge_preunsmear_logs/unmerge_preunsmear_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

LOG_DIR=${LOG_DIR:-"unmerge_preunsmear_logs"}
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Unmerge Pre-Unsmear Pipeline (single)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

source ~/.bashrc
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
  python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_preunsmear_sweep"}
RUN_NAME=${RUN_NAME:-"preunsmear_base"}
UNMERGE_LOSS=${UNMERGE_LOSS:-"chamfer"}
NUM_WORKERS=${NUM_WORKERS:-6}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
NO_CURRICULUM=${NO_CURRICULUM:-0}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.0}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-1}
K_FOLDS=${K_FOLDS:-1}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-0}
KFOLD_USE_PRETRAINED=${KFOLD_USE_PRETRAINED:-0}
KFOLD_FIXED_FOLD=${KFOLD_FIXED_FOLD:--1}
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/kfold_models"}
MC_SWEEP=${MC_SWEEP:-0}
CLASSIFIER_ONLY=${CLASSIFIER_ONLY:-0}
UNMERGED_CACHE=${UNMERGED_CACHE:-""}
UNSMEAR_K_FOLDS=${UNSMEAR_K_FOLDS:-1}
UNSMEAR_KFOLD_MODEL_DIR=${UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/unsmear_kfold_models"}
UNSMEAR_KFOLD_USE_PRETRAINED=${UNSMEAR_KFOLD_USE_PRETRAINED:-0}
UNSMEAR_KFOLD_ENSEMBLE=${UNSMEAR_KFOLD_ENSEMBLE:-0}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
SKIP_BASELINE=${SKIP_BASELINE:-0}
UNSMEAR_MERGE_FLAG=${UNSMEAR_MERGE_FLAG:-0}
UNSMEAR_MERGE_WEIGHT=${UNSMEAR_MERGE_WEIGHT:-2.0}
UNSMEAR_TWO_HEAD=${UNSMEAR_TWO_HEAD:-0}
UNSMEAR_LOCAL_ATTN=${UNSMEAR_LOCAL_ATTN:-0}
UNSMEAR_LOCAL_ATTN_RADIUS=${UNSMEAR_LOCAL_ATTN_RADIUS:-0.2}
UNSMEAR_LOCAL_ATTN_SCALE=${UNSMEAR_LOCAL_ATTN_SCALE:-2.0}
UNSMEAR_FEAT_HEAD_MODE=${UNSMEAR_FEAT_HEAD_MODE:-"single"}
STOP_AFTER_UNMERGE=${STOP_AFTER_UNMERGE:-0}
PRE_UNSMEAR=${PRE_UNSMEAR:-0}
PRE_UNSMEAR_WEIGHT=${PRE_UNSMEAR_WEIGHT:-2.0}
PRE_UNSMEAR_K_FOLDS=${PRE_UNSMEAR_K_FOLDS:-1}
PRE_UNSMEAR_KFOLD_MODEL_DIR=${PRE_UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/pre_unsmear_models"}
PRE_UNSMEAR_KFOLD_USE_PRETRAINED=${PRE_UNSMEAR_KFOLD_USE_PRETRAINED:-0}
PRE_UNSMEAR_KFOLD_ENSEMBLE=${PRE_UNSMEAR_KFOLD_ENSEMBLE:-0}
UNMERGE_RELPOS_MODE=${UNMERGE_RELPOS_MODE:-"none"}

CMD="python unmerge_distr_model_unsmear.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --unmerge_loss $UNMERGE_LOSS \
  --num_workers $NUM_WORKERS \
  --curriculum_start $CURR_START \
  --curriculum_epochs $CURR_EPOCHS \
  --physics_weight $PHYSICS_WEIGHT \
  --nll_weight $NLL_WEIGHT \
  --k_folds $K_FOLDS \
  --kfold_model_dir $KFOLD_MODEL_DIR \
  --save_unmerged_cache"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$USE_TRUE_COUNT" -eq 1 ]; then
  CMD="$CMD --use_true_count"
fi

if [ "$KFOLD_ENSEMBLE" -eq 1 ]; then
  CMD="$CMD --kfold_ensemble_valtest"
fi

if [ "$KFOLD_USE_PRETRAINED" -eq 1 ]; then
  CMD="$CMD --kfold_use_pretrained"
fi

if [ "$KFOLD_FIXED_FOLD" -ge 0 ]; then
  CMD="$CMD --kfold_fixed_fold $KFOLD_FIXED_FOLD"
fi

if [ "$NO_CURRICULUM" -eq 1 ]; then
  CMD="$CMD --no_curriculum"
fi

if [ "$NO_DISTRIBUTIONAL" -eq 1 ]; then
  CMD="$CMD --no_distributional"
fi

if [ "$CLASSIFIER_ONLY" -eq 1 ]; then
  CMD="$CMD --classifier_only"
fi

if [ "$STOP_AFTER_UNMERGE" -eq 1 ]; then
  CMD="$CMD --stop_after_unmerge --save_unmerged_cache"
fi

if [ -n "$UNMERGED_CACHE" ]; then
  CMD="$CMD --load_unmerged_cache $UNMERGED_CACHE"
fi

if [ "$UNSMEAR_K_FOLDS" -gt 1 ]; then
  CMD="$CMD --unsmear_k_folds $UNSMEAR_K_FOLDS"
fi

if [ -n "$UNSMEAR_KFOLD_MODEL_DIR" ]; then
  CMD="$CMD --unsmear_kfold_model_dir $UNSMEAR_KFOLD_MODEL_DIR"
fi

if [ "$UNSMEAR_KFOLD_USE_PRETRAINED" -eq 1 ]; then
  CMD="$CMD --unsmear_kfold_use_pretrained"
fi

if [ "$UNSMEAR_KFOLD_ENSEMBLE" -eq 1 ]; then
  CMD="$CMD --unsmear_kfold_ensemble_valtest"
fi

if [ "$UNSMEAR_MERGE_FLAG" -eq 1 ]; then
  CMD="$CMD --unsmear_merge_flag --unsmear_merge_weight $UNSMEAR_MERGE_WEIGHT"
fi

if [ "$UNSMEAR_TWO_HEAD" -eq 1 ]; then
  CMD="$CMD --unsmear_two_head"
fi

if [ "$UNSMEAR_LOCAL_ATTN" -eq 1 ]; then
  CMD="$CMD --unsmear_local_attn --unsmear_local_attn_radius $UNSMEAR_LOCAL_ATTN_RADIUS --unsmear_local_attn_scale $UNSMEAR_LOCAL_ATTN_SCALE"
fi

if [ -n "$UNSMEAR_FEAT_HEAD_MODE" ]; then
  CMD="$CMD --unsmear_feat_head_mode $UNSMEAR_FEAT_HEAD_MODE"
fi

if [ "$PRE_UNSMEAR" -eq 1 ]; then
  CMD="$CMD --pre_unsmear_singletons --pre_unsmear_weight $PRE_UNSMEAR_WEIGHT --pre_unsmear_k_folds $PRE_UNSMEAR_K_FOLDS --pre_unsmear_kfold_model_dir $PRE_UNSMEAR_KFOLD_MODEL_DIR"
  if [ "$PRE_UNSMEAR_KFOLD_USE_PRETRAINED" -eq 1 ]; then
    CMD="$CMD --pre_unsmear_kfold_use_pretrained"
  fi
  if [ "$PRE_UNSMEAR_KFOLD_ENSEMBLE" -eq 1 ]; then
    CMD="$CMD --pre_unsmear_kfold_ensemble_valtest"
  fi
fi

if [ -n "$TEACHER_CKPT" ]; then
  CMD="$CMD --teacher_checkpoint $TEACHER_CKPT"
fi

if [ -n "$BASELINE_CKPT" ]; then
  CMD="$CMD --baseline_checkpoint $BASELINE_CKPT"
fi

if [ "$SKIP_BASELINE" -eq 1 ]; then
  CMD="$CMD --skip_baseline"
fi

if [ "$UNMERGE_RELPOS_MODE" != "none" ]; then
  CMD="$CMD --unmerge_relpos_mode $UNMERGE_RELPOS_MODE"
fi

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CMD="$CMD --device cuda"
else
  CMD="$CMD --device cpu"
fi

echo "Running command:"
echo "$CMD"

eval "$CMD"
