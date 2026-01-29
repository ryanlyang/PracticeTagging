#!/bin/bash

# K-fold pre-unsmearer training job (train-only), runs before unmerger.

#SBATCH --job-name=pre_unsmear_kf
## Set log directory (can override with LOG_DIR env)
#SBATCH --output=unmerge_distr_kfold_unsmear_logs/pre_unsmear_kfold_fold_%j.out
#SBATCH --error=unmerge_distr_kfold_unsmear_logs/pre_unsmear_kfold_fold_%j.err
#SBATCH --time=11-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

LOG_DIR=${LOG_DIR:-"unmerge_distr_kfold_unsmear_logs"}
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Pre-Unsmearer K-fold (train-only)"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep"}
RUN_NAME=${RUN_NAME:-"kfold_pre_unsmear"}
NUM_WORKERS=${NUM_WORKERS:-6}
PRE_UNSMEAR_K_FOLDS=${PRE_UNSMEAR_K_FOLDS:-5}
PRE_UNSMEAR_KFOLD_MODEL_DIR=${PRE_UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/pre_unsmear_kfold_models"}
PRE_UNSMEAR_WEIGHT=${PRE_UNSMEAR_WEIGHT:-2.0}
FOLD_ID=${FOLD_ID:-0}

CMD="python unmerge_distr_model_unsmear.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --num_workers $NUM_WORKERS \
  --pre_unsmear_singletons \
  --pre_unsmear_weight $PRE_UNSMEAR_WEIGHT \
  --pre_unsmear_k_folds $PRE_UNSMEAR_K_FOLDS \
  --pre_unsmear_kfold_train_only $FOLD_ID \
  --pre_unsmear_kfold_model_dir $PRE_UNSMEAR_KFOLD_MODEL_DIR"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
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
  echo "Pre-unsmear fold completed successfully"
  echo "Model saved to: $PRE_UNSMEAR_KFOLD_MODEL_DIR/fold_$FOLD_ID"
else
  echo "Pre-unsmear fold failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="
exit $EXIT_CODE
