#!/bin/bash

#SBATCH --job-name=unmerge_new
#SBATCH --output=unmerge_new_ideas_logs/unmerge_new_%j.out
#SBATCH --error=unmerge_new_ideas_logs/unmerge_new_%j.err
#SBATCH --time=8-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_new_ideas_logs

echo "=========================================="
echo "Unmerge New Ideas"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_new"}
RUN_NAME=${RUN_NAME:-"default"}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

UNMERGE_HEAD_MODE=${UNMERGE_HEAD_MODE:-"single"}
UNMERGE_PARENT_MODE=${UNMERGE_PARENT_MODE:-"none"}
UNMERGE_RELPOS_MODE=${UNMERGE_RELPOS_MODE:-"none"}
UNMERGE_LOCAL_ATTN_MODE=${UNMERGE_LOCAL_ATTN_MODE:-"none"}
UNMERGE_LOCAL_ATTN_RADIUS=${UNMERGE_LOCAL_ATTN_RADIUS:-0.2}
UNMERGE_LOCAL_ATTN_SCALE=${UNMERGE_LOCAL_ATTN_SCALE:-2.0}
UNMERGE_TARGET_MODE=${UNMERGE_TARGET_MODE:-"absolute"}
UNMERGE_COUNT_BALANCED=${UNMERGE_COUNT_BALANCED:-0}

CMD="python unmerge_new_ideas.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --unmerge_head_mode $UNMERGE_HEAD_MODE \
  --unmerge_parent_mode $UNMERGE_PARENT_MODE \
  --unmerge_relpos_mode $UNMERGE_RELPOS_MODE \
  --unmerge_local_attn_mode $UNMERGE_LOCAL_ATTN_MODE \
  --unmerge_local_attn_radius $UNMERGE_LOCAL_ATTN_RADIUS \
  --unmerge_local_attn_scale $UNMERGE_LOCAL_ATTN_SCALE \
  --unmerge_target_mode $UNMERGE_TARGET_MODE"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$UNMERGE_COUNT_BALANCED" -eq 1 ]; then
  CMD="$CMD --unmerge_count_balanced"
fi

if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
  CMD="$CMD --skip_save_models"
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
  echo "Run completed successfully"
  echo "Results saved to: $SAVE_DIR/$RUN_NAME"
else
  echo "Run failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
