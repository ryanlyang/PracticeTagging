#!/bin/bash

#SBATCH --job-name=unmerge
#SBATCH --output=unmerge_logs/unmerge_%j.out
#SBATCH --error=unmerge_logs/unmerge_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_logs

echo "=========================================="
echo "Unmerge Model - Debug"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerging"}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
PATIENCE=${PATIENCE:-20}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}
N_PRINT_EXAMPLES=${N_PRINT_EXAMPLES:-5}

BASE_CMD="python unmerging.py \
  --save_dir $SAVE_DIR \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --warmup_epochs $WARMUP_EPOCHS \
  --patience $PATIENCE \
  --n_print_examples $N_PRINT_EXAMPLES"

if [ -n "$TRAIN_PATH" ]; then
  BASE_CMD="$BASE_CMD --train_path $TRAIN_PATH"
fi

if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
  BASE_CMD="$BASE_CMD --skip_save_models"
fi

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  BASE_CMD="$BASE_CMD --device cuda"
else
  BASE_CMD="$BASE_CMD --device cpu"
fi

echo "=========================================="
echo "Run 1: smearing on (pt/eta/phi)"
echo "=========================================="
CMD1="$BASE_CMD --run_name smear_pt_eta_phi --pt_resolution 0.10 --eta_resolution 0.03 --phi_resolution 0.03"
echo "$CMD1"
eval $CMD1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Run 1 failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Run 2: default config (current)"
echo "=========================================="
CMD2="$BASE_CMD --run_name default_config"
echo "$CMD2"
eval $CMD2

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "All runs completed successfully"
  echo "Results saved to: $SAVE_DIR"
else
  echo "Run 2 failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
