#!/bin/bash

# Sbatch runner for DualView+KD-only retrain on a fresh 200k slice.

#SBATCH --job-name=unmerge_dv_kd
#SBATCH --output=unmerge_new_dualview_kd_logs/unmerge_dv_kd_%j.out
#SBATCH --error=unmerge_new_dualview_kd_logs/unmerge_dv_kd_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_new_dualview_kd_logs

echo "=========================================="
echo "Unmerge DualView+KD (new 200k slice)"
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

CKPT_DIR=${CKPT_DIR:-"checkpoints/unmerge_new/relpos"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_new/relpos_newdata"}
RUN_NAME=${RUN_NAME:-"dualview_kd_new200k"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
OFFSET_JETS=${OFFSET_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}

UNMERGE_HEAD_MODE=${UNMERGE_HEAD_MODE:-"single"}
UNMERGE_PARENT_MODE=${UNMERGE_PARENT_MODE:-"none"}
UNMERGE_RELPOS_MODE=${UNMERGE_RELPOS_MODE:-"none"}
UNMERGE_LOCAL_ATTN_MODE=${UNMERGE_LOCAL_ATTN_MODE:-"none"}
UNMERGE_LOCAL_ATTN_RADIUS=${UNMERGE_LOCAL_ATTN_RADIUS:-0.2}
UNMERGE_LOCAL_ATTN_SCALE=${UNMERGE_LOCAL_ATTN_SCALE:-2.0}
UNMERGE_TARGET_MODE=${UNMERGE_TARGET_MODE:-"absolute"}
KD_SWEEP=${KD_SWEEP:-1}
KD_SWEEP_MAX=${KD_SWEEP_MAX:-30}
KD_SWEEP_TARGET=${KD_SWEEP_TARGET:-"dual_flag"}

CMD="python run_unmerge_new_dualview_kd_only.py \
  --ckpt_dir $CKPT_DIR \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --offset_jets $OFFSET_JETS \
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

if [ "$KD_SWEEP" -eq 1 ]; then
  CMD="$CMD --kd_sweep --kd_sweep_max $KD_SWEEP_MAX --kd_sweep_target $KD_SWEEP_TARGET"
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
