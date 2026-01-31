#!/bin/bash

# Retrain classifiers on a NEW 200k-jet subset using a pretrained unmerger + merge-count predictor.

#SBATCH --job-name=unmerge_retrain_new
#SBATCH --output=unmerge_retrain_newdata_logs/unmerge_retrain_new_%j.out
#SBATCH --error=unmerge_retrain_newdata_logs/unmerge_retrain_new_%j.err
#SBATCH --time=4-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unmerge_retrain_newdata_logs

echo "=========================================="
echo "Unmerge Retrain on New Jets"
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

CKPT_DIR=${CKPT_DIR:-"checkpoints/unmerge_model/default"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_model_retrain"}
RUN_NAME=${RUN_NAME:-"retrain_newdata_offset200k"}

N_EVAL_JETS=${N_EVAL_JETS:-200000}
OFFSET_JETS=${OFFSET_JETS:-200000}
STATS_JETS=${STATS_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
NUM_WORKERS=${NUM_WORKERS:-6}

CMD="python unmerge_model_retrain_newdata.py \
  --ckpt_dir $CKPT_DIR \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_eval_jets $N_EVAL_JETS \
  --offset_jets $OFFSET_JETS \
  --stats_jets $STATS_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --num_workers $NUM_WORKERS"

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
  echo "Pipeline completed successfully"
  echo "Results saved to: $SAVE_DIR/$RUN_NAME"
else
  echo "Pipeline failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
