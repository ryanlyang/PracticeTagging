#!/bin/bash

#SBATCH --job-name=unmerge_noise
#SBATCH --output=unmerge_noise_logs/unmerge_noise_%j.out
#SBATCH --error=unmerge_noise_logs/unmerge_noise_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=debug
#SBATCH --gres=gpu:1

mkdir -p unmerge_noise_logs

echo "=========================================="
echo "Unmerge Noise Injection - Debug"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unmerge_model_noise"}
RUN_NAME=${RUN_NAME:-"noise_v1"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/unmerge_model/default"}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

# Noise knobs (defaults match script)
DROP_PROB=${DROP_PROB:-0.05}
PT_SCALE=${PT_SCALE:-0.10}
PT_JITTER=${PT_JITTER:-0.0}
ETA_JITTER=${ETA_JITTER:-0.01}
PHI_JITTER=${PHI_JITTER:-0.01}
ENERGY_JITTER=${ENERGY_JITTER:-0.0}
NO_RECOMPUTE_E=${NO_RECOMPUTE_E:-0}
NOISE_SEED=${NOISE_SEED:-1337}

CMD="python unmerge_noise_injection.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --ckpt_dir $CKPT_DIR \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --max_merge_count $MAX_MERGE_COUNT \
  --drop_prob $DROP_PROB \
  --pt_scale $PT_SCALE \
  --pt_jitter $PT_JITTER \
  --eta_jitter $ETA_JITTER \
  --phi_jitter $PHI_JITTER \
  --energy_jitter $ENERGY_JITTER \
  --noise_seed $NOISE_SEED"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$NO_RECOMPUTE_E" -eq 1 ]; then
  CMD="$CMD --no_recompute_E"
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
