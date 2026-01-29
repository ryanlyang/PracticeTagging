#!/bin/bash

# K-fold unsmearer training job (train-only), using cached unmerged dataset.

#SBATCH --job-name=unsmear_kf
#SBATCH --output=unmerge_distr_kfold_logs/unsmear_kfold_fold_%j.out
#SBATCH --error=unmerge_distr_kfold_logs/unsmear_kfold_fold_%j.err
#SBATCH --time=11-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_distr_kfold_logs

echo "=========================================="
echo "Unsmearer K-fold (train-only)"
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

UNMERGED_CACHE=${UNMERGED_CACHE:-""}
if [ -z "$UNMERGED_CACHE" ]; then
  echo "ERROR: UNMERGED_CACHE is required for unsmear k-fold training."
  exit 1
fi

N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold_unsmear"}
RUN_NAME=${RUN_NAME:-"kfold_unsmear"}
NUM_WORKERS=${NUM_WORKERS:-6}
UNSMEAR_K_FOLDS=${UNSMEAR_K_FOLDS:-5}
UNSMEAR_KFOLD_MODEL_DIR=${UNSMEAR_KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/unsmear_kfold_models"}
FOLD_ID=${FOLD_ID:-0}
UNSMEAR_MERGE_FLAG=${UNSMEAR_MERGE_FLAG:-0}
UNSMEAR_MERGE_WEIGHT=${UNSMEAR_MERGE_WEIGHT:-2.0}
UNSMEAR_TWO_HEAD=${UNSMEAR_TWO_HEAD:-0}

CMD="python unmerge_distr_model_unsmear.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --num_workers $NUM_WORKERS \
  --load_unmerged_cache $UNMERGED_CACHE \
  --unsmear_k_folds $UNSMEAR_K_FOLDS \
  --unsmear_kfold_train_only $FOLD_ID \
  --unsmear_kfold_model_dir $UNSMEAR_KFOLD_MODEL_DIR \
  --device cuda"

if [ "$UNSMEAR_MERGE_FLAG" -eq 1 ]; then
  CMD="$CMD --unsmear_merge_flag --unsmear_merge_weight $UNSMEAR_MERGE_WEIGHT"
fi
if [ "$UNSMEAR_TWO_HEAD" -eq 1 ]; then
  CMD="$CMD --unsmear_two_head"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Unsmear fold completed successfully"
else
  echo "Unsmear fold failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="
exit $EXIT_CODE
