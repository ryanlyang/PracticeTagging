#!/bin/bash

#SBATCH --job-name=unmerge_distr_final
#SBATCH --output=unmerge_distr_kfold_logs/unmerge_distr_final_%j.out
#SBATCH --error=unmerge_distr_kfold_logs/unmerge_distr_final_%j.err
#SBATCH --time=4-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_distr_kfold_logs

echo "=========================================="
echo "Unmerge Distributional Model (K-fold final)"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_distr_kfold"}
RUN_NAME=${RUN_NAME:-"kfold_run"}
UNMERGE_LOSS=${UNMERGE_LOSS:-"hungarian"}
USE_TRUE_COUNT=${USE_TRUE_COUNT:-0}
NO_CURRICULUM=${NO_CURRICULUM:-0}
CURR_START=${CURR_START:-2}
CURR_EPOCHS=${CURR_EPOCHS:-20}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.2}
NLL_WEIGHT=${NLL_WEIGHT:-1.0}
NO_DISTRIBUTIONAL=${NO_DISTRIBUTIONAL:-0}
K_FOLDS=${K_FOLDS:-5}
KFOLD_ENSEMBLE=${KFOLD_ENSEMBLE:-1}
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"$SAVE_DIR/$RUN_NAME/kfold_models"}

CMD="python unmerge_distr_model.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --unmerge_loss $UNMERGE_LOSS \
  --curriculum_start $CURR_START \
  --curriculum_epochs $CURR_EPOCHS \
  --physics_weight $PHYSICS_WEIGHT \
  --nll_weight $NLL_WEIGHT \
  --k_folds $K_FOLDS \
  --kfold_model_dir $KFOLD_MODEL_DIR \
  --kfold_use_pretrained \
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

if [ "$NO_CURRICULUM" -eq 1 ]; then
  CMD="$CMD --no_curriculum"
fi

if [ "$NO_DISTRIBUTIONAL" -eq 1 ]; then
  CMD="$CMD --no_distributional"
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
  echo "Final run completed successfully"
  echo "Results saved to: $SAVE_DIR/$RUN_NAME"
else
  echo "Final run failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
