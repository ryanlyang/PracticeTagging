#!/bin/bash

#SBATCH --job-name=unsmear_cls
#SBATCH --output=unsmear_logs/unsmear_cls_%j.out
#SBATCH --error=unsmear_logs/unsmear_cls_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p unsmear_logs

echo "=========================================="
echo "Unsmear Classifiers-Only"
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

SAVE_DIR=${SAVE_DIR:-"checkpoints/unsmear"}
RUN_NAME=${RUN_NAME:-"default_classifiers"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
RESULTS_NPZ=${RESULTS_NPZ:-"checkpoints/unsmear/default/results.npz"}
DIFF_CKPT=${DIFF_CKPT:-"checkpoints/unsmear/default/unsmear_diffusion_ema.pt"}

CMD="python unsmear_method.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --classifiers_only \
  --results_npz $RESULTS_NPZ \
  --diffusion_ckpt $DIFF_CKPT"

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
