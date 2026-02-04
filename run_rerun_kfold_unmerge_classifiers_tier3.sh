#!/bin/bash
#
# Rerun classifier training/eval from pretrained k-fold merge-count + unmerge predictors.
# This is meant to be run on the cluster with Slurm.
#

#SBATCH --job-name=rerun_kfold_unmerge
#SBATCH --output=rerun_kfold_unmerge_logs/rerun_kfold_unmerge_%j.out
#SBATCH --error=rerun_kfold_unmerge_logs/rerun_kfold_unmerge_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p rerun_kfold_unmerge_logs

echo "=========================================="
echo "Rerun K-fold Unmerge Classifiers"
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

TRAIN_PATH=${TRAIN_PATH:-"./data"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
OFFSET_JETS=${OFFSET_JETS:-0}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
NUM_WORKERS=${NUM_WORKERS:-6}
SAVE_DIR=${SAVE_DIR:-"checkpoints/rerun_kfold_unmerge"}
RUN_NAME=${RUN_NAME:-"rerun"}
EFFICIENCY_LOSS=${EFFICIENCY_LOSS:-0.05}
RANDOM_VALTEST=${RANDOM_VALTEST:-1}

# REQUIRED: path containing fold_{0..K-1}/merge_count.pt and unmerge_predictor.pt
KFOLD_MODEL_DIR=${KFOLD_MODEL_DIR:-"checkpoints/unmerge_distr_kfold_unsmear_sweep/kfold_base_det_unsmear/kfold_models"}
K_FOLDS=${K_FOLDS:-5}

CMD="python rerun_kfold_unmerge_classifiers.py \
  --train_path $TRAIN_PATH \
  --n_train_jets $N_TRAIN_JETS \
  --offset_jets $OFFSET_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --k_folds $K_FOLDS \
  --kfold_model_dir $KFOLD_MODEL_DIR \
  --num_workers $NUM_WORKERS \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --efficiency_loss $EFFICIENCY_LOSS"

if [ "$RANDOM_VALTEST" -eq 1 ]; then
  CMD="$CMD --random_valtest"
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

