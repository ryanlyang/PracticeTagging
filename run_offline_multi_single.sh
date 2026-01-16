#!/bin/bash

#SBATCH --job-name=offline_multi
#SBATCH --output=offline_multi_logs/offline_multi_%j.out
#SBATCH --error=offline_multi_logs/offline_multi_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

# Create log directory
mkdir -p offline_multi_logs

# Print job info
echo "=========================================="
echo "Offline Sampler A2 (HLT Student)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate atlas_kd

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Print environment
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Hyperparameter values (passed from submission script)
KD_TEMP=${KD_TEMP:-4.0}
A_KD=${A_KD:-1.0}
A_SUP=${A_SUP:-1.0}
A_EMB=${A_EMB:-1.0}
BETA_ALL=${BETA_ALL:-0.1}
TAU_U=${TAU_U:-0.05}
W_MIN=${W_MIN:-0.1}
K_SAMPLES=${K_SAMPLES:-16}

GEN_EPOCHS=${GEN_EPOCHS:-30}
GEN_LR=${GEN_LR:-1e-3}
GEN_LAMBDA_MASK=${GEN_LAMBDA_MASK:-1.0}
GEN_LAMBDA_PERC=${GEN_LAMBDA_PERC:-1.0}
GEN_LAMBDA_LOGIT=${GEN_LAMBDA_LOGIT:-0.5}
GEN_MIN_SIGMA=${GEN_MIN_SIGMA:-0.02}
GEN_GLOBAL_NOISE=${GEN_GLOBAL_NOISE:-0.05}

RUN_NAME=${RUN_NAME:-"default"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/offline_multi_sweep"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
GENERATOR_CKPT=${GENERATOR_CKPT:-""}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  kd_temp: $KD_TEMP"
echo "  a_sup/a_kd/a_emb: $A_SUP / $A_KD / $A_EMB"
echo "  beta_all: $BETA_ALL"
echo "  tau_u: $TAU_U"
echo "  w_min: $W_MIN"
echo "  k_samples: $K_SAMPLES"
echo "  gen_min_sigma: $GEN_MIN_SIGMA"
echo "  gen_lambda_mask: $GEN_LAMBDA_MASK"
echo "  gen_lambda_perc: $GEN_LAMBDA_PERC"
echo "  gen_lambda_logit: $GEN_LAMBDA_LOGIT"
echo ""

CMD="python Offline_multi.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --kd_temp $KD_TEMP \
    --a_sup $A_SUP \
    --a_kd $A_KD \
    --a_emb $A_EMB \
    --beta_all $BETA_ALL \
    --tau_u $TAU_U \
    --w_min $W_MIN \
    --k_samples $K_SAMPLES \
    --gen_epochs $GEN_EPOCHS \
    --gen_lr $GEN_LR \
    --gen_lambda_mask $GEN_LAMBDA_MASK \
    --gen_lambda_perc $GEN_LAMBDA_PERC \
    --gen_lambda_logit $GEN_LAMBDA_LOGIT \
    --gen_min_sigma $GEN_MIN_SIGMA \
    --gen_global_noise $GEN_GLOBAL_NOISE"

if [ -n "$TRAIN_PATH" ]; then
    CMD="$CMD --train_path $TRAIN_PATH"
fi

if [ -n "$N_TRAIN_JETS" ]; then
    CMD="$CMD --n_train_jets $N_TRAIN_JETS"
fi

if [ -n "$MAX_CONSTITS" ]; then
    CMD="$CMD --max_constits $MAX_CONSTITS"
fi

if [ -n "$TEACHER_CKPT" ]; then
    CMD="$CMD --teacher_checkpoint $TEACHER_CKPT"
fi

if [ -n "$BASELINE_CKPT" ]; then
    CMD="$CMD --baseline_checkpoint $BASELINE_CKPT"
fi

if [ -n "$GENERATOR_CKPT" ]; then
    CMD="$CMD --generator_checkpoint $GENERATOR_CKPT"
fi

if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
    CMD="$CMD --skip_save_models"
fi

# Check if GPU is available and add device flag
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CMD="$CMD --device cuda"
    echo "Using GPU for training"
else
    CMD="$CMD --device cpu"
    echo "Using CPU for training"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    echo "Results saved to: $SAVE_DIR/$RUN_NAME"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
