#!/bin/bash

# K-fold unmerger training job (unsmear pipeline variant)

#SBATCH --job-name=unmerge_distr_fu
## Set log directory (can override with LOG_DIR env)
#SBATCH --output=unmerge_distr_kfold_unsmear_logs/unmerge_distr_fold_unsmear_%j.out
#SBATCH --error=unmerge_distr_kfold_unsmear_logs/unmerge_distr_fold_unsmear_%j.err
#SBATCH --time=11-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

LOG_DIR=${LOG_DIR:-"unmerge_distr_kfold_unsmear_logs"}
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Unmerge Distributional Model (K-fold train-only, unsmear pipeline)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Fold: ${FOLD_ID:-0}"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear"}
RUN_NAME=${RUN_NAME:-"kfold_unsmear"}
UNMERGE_LOSS=${UNMERGE_LOSS:-"hungarian"}
NUM_WORKERS=${NUM_WORKERS:-6}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
NO_CURRICULUM=${NO_CURRICULUM:-0}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.2}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-0}
K_FOLDS=${K_FOLDS:-5}
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/kfold_models"}
FOLD_ID=${FOLD_ID:-0}
PRE_UNSMEAR=${PRE_UNSMEAR:-0}
PRE_UNSMEAR_WEIGHT=${PRE_UNSMEAR_WEIGHT:-2.0}
PRE_UNSMEAR_K_FOLDS=${PRE_UNSMEAR_K_FOLDS:-$K_FOLDS}
PRE_UNSMEAR_KFOLD_MODEL_DIR=${PRE_UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/pre_unsmear_kfold_models"}
PRE_UNSMEAR_KFOLD_USE_PRETRAINED=${PRE_UNSMEAR_KFOLD_USE_PRETRAINED:-0}
PRE_UNSMEAR_KFOLD_ENSEMBLE=${PRE_UNSMEAR_KFOLD_ENSEMBLE:-1}

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
  --kfold_train_only $FOLD_ID \
  --kfold_model_dir $KFOLD_MODEL_DIR"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$USE_TRUE_COUNT" -eq 1 ]; then
  CMD="$CMD --use_true_count"
fi

if [ "$NO_CURRICULUM" -eq 1 ]; then
  CMD="$CMD --no_curriculum"
fi

if [ "$NO_DISTRIBUTIONAL" -eq 1 ]; then
  CMD="$CMD --no_distributional"
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
  echo "Fold run completed successfully"
  echo "Fold models saved to: $KFOLD_MODEL_DIR/fold_$FOLD_ID"
else
  echo "Fold run failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
