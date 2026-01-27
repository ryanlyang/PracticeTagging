#!/bin/bash

#SBATCH --job-name=unsmear
#SBATCH --output=unsmear_logs/unsmear_%j.out
#SBATCH --error=unsmear_logs/unsmear_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=debug
#SBATCH --gres=gpu:1

mkdir -p unsmear_logs

echo "=========================================="
echo "Unsmear Diffusion + Classifiers"
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
SAVE_DIR=${SAVE_DIR:-"checkpoints/unsmear"}
RUN_NAME=${RUN_NAME:-"default"}

PRED_TYPE=${PRED_TYPE:-"v"}
SNR_WEIGHT=${SNR_WEIGHT:-1}
SNR_GAMMA=${SNR_GAMMA:-5.0}
SELF_COND_PROB=${SELF_COND_PROB:-0.5}
COND_DROP_PROB=${COND_DROP_PROB:-0.1}
JET_LOSS_WEIGHT=${JET_LOSS_WEIGHT:-0.1}
USE_CROSS_ATTN=${USE_CROSS_ATTN:-1}
NO_SELF_COND=${NO_SELF_COND:-0}
SAMPLING_METHOD=${SAMPLING_METHOD:-"ddim"}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.5}
SAMPLE_STEPS=${SAMPLE_STEPS:-400}
N_SAMPLES_EVAL=${N_SAMPLES_EVAL:-4}
SKIP_CLASSIFIERS=${SKIP_CLASSIFIERS:-0}

CMD="python unsmear_method.py \
  --save_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --n_train_jets $N_TRAIN_JETS \
  --max_constits $MAX_CONSTITS \
  --pred_type $PRED_TYPE \
  --snr_gamma $SNR_GAMMA \
  --self_cond_prob $SELF_COND_PROB \
  --cond_drop_prob $COND_DROP_PROB \
  --jet_loss_weight $JET_LOSS_WEIGHT \
  --sampling_method $SAMPLING_METHOD \
  --guidance_scale $GUIDANCE_SCALE \
  --sample_steps $SAMPLE_STEPS \
  --n_samples_eval $N_SAMPLES_EVAL"

if [ -n "$TRAIN_PATH" ]; then
  CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ "$SNR_WEIGHT" -eq 1 ]; then
  CMD="$CMD --snr_weight"
fi

if [ "$USE_CROSS_ATTN" -eq 1 ]; then
  CMD="$CMD --use_cross_attn"
fi

if [ "$NO_SELF_COND" -eq 1 ]; then
  CMD="$CMD --no_self_cond"
fi

if [ "$SKIP_CLASSIFIERS" -eq 1 ]; then
  CMD="$CMD --skip_classifiers"
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
