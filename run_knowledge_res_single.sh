#!/bin/bash

#SBATCH --job-name=knowledge_res
#SBATCH --output=knowledge_res_logs/knowledge_res_%j.out
#SBATCH --error=knowledge_res_logs/knowledge_res_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p knowledge_res_logs

echo "=========================================="
echo "KnowledgeRes (HLT -> Offline Distribution)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

source ~/.bashrc
conda activate atlas_kd

cd $SLURM_SUBMIT_DIR

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

RUN_NAME=${RUN_NAME:-"default"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/knowledge_res"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-100000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

RUN_TEACHER=${RUN_TEACHER:-0}
RUN_BASELINE=${RUN_BASELINE:-0}
RUN_STUDENT=${RUN_STUDENT:-1}
USE_KD=${USE_KD:-0}

TEACHER_CKPT=${TEACHER_CKPT:-""}
BASELINE_CKPT=${BASELINE_CKPT:-""}
STUDENT_CKPT=${STUDENT_CKPT:-""}

# Compare checkpoints (for ROC plot)
COMPARE_TEACHER_CKPT=${COMPARE_TEACHER_CKPT:-"/home/ryreu/atlas/PracticeTagging/checkpoints/smartres_sweep/A_K4_L4_H128/teacher.pt"}
COMPARE_BASELINE_CKPT=${COMPARE_BASELINE_CKPT:-"/home/ryreu/atlas/PracticeTagging/checkpoints/smartres_sweep/A_K4_L4_H128/baseline.pt"}
COMPARE_STUDENT_CKPT=${COMPARE_STUDENT_CKPT:-""}

# Knowledge sampler
KNOW_SAMPLES=${KNOW_SAMPLES:-128}
KNOW_EVAL_SAMPLES=${KNOW_EVAL_SAMPLES:-128}
INV_NOISE_SCALE=${INV_NOISE_SCALE:-1.0}
EXTRA_COUNT_SCALE=${EXTRA_COUNT_SCALE:-1.0}
SPLIT_FRAC=${SPLIT_FRAC:-0.5}
SPLIT_RADIUS=${SPLIT_RADIUS:-0.01}
SPLIT_MIN_FRAC=${SPLIT_MIN_FRAC:-0.05}
SPLIT_MAX_FRAC=${SPLIT_MAX_FRAC:-0.3}
CONSERVE_PT=${CONSERVE_PT:-1}

# Locked KD/consistency hyperparameters
CONF_POWER=${CONF_POWER:-2.0}
CONF_MIN=${CONF_MIN:-0.0}
TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}
ALPHA_ATTN=${ALPHA_ATTN:-0.05}
ALPHA_REP=${ALPHA_REP:-0.10}
ALPHA_NCE=${ALPHA_NCE:-0.10}
TAU_NCE=${TAU_NCE:-0.10}
CONF_WEIGHTED_KD=${CONF_WEIGHTED_KD:-1}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  run_teacher/run_baseline/run_student: $RUN_TEACHER / $RUN_BASELINE / $RUN_STUDENT"
echo "  use_kd: $USE_KD"
echo "  knowledge_samples (train/eval): $KNOW_SAMPLES / $KNOW_EVAL_SAMPLES"
echo ""

CMD="python KnowledgeRes.py \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --knowledge_samples $KNOW_SAMPLES \
    --knowledge_eval_samples $KNOW_EVAL_SAMPLES \
    --inv_noise_scale $INV_NOISE_SCALE \
    --extra_count_scale $EXTRA_COUNT_SCALE \
    --split_frac $SPLIT_FRAC \
    --split_radius $SPLIT_RADIUS \
    --split_min_frac $SPLIT_MIN_FRAC \
    --split_max_frac $SPLIT_MAX_FRAC \
    --conf_power $CONF_POWER \
    --conf_min $CONF_MIN \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT \
    --alpha_attn $ALPHA_ATTN \
    --alpha_rep $ALPHA_REP \
    --alpha_nce $ALPHA_NCE \
    --tau_nce $TAU_NCE"

if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi
if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
fi
if [ "$CONF_WEIGHTED_KD" -eq 0 ]; then
    CMD="$CMD --no_conf_kd"
fi
if [ "$CONSERVE_PT" -eq 0 ]; then
    CMD="$CMD --no_conserve_pt"
fi
if [ -n "$COMPARE_STUDENT_CKPT" ]; then
    CMD="$CMD --compare_student_checkpoint $COMPARE_STUDENT_CKPT"
fi
if [ -n "$COMPARE_TEACHER_CKPT" ]; then
    CMD="$CMD --compare_teacher_checkpoint $COMPARE_TEACHER_CKPT"
fi
if [ -n "$COMPARE_BASELINE_CKPT" ]; then
    CMD="$CMD --compare_baseline_checkpoint $COMPARE_BASELINE_CKPT"
fi

if [ "$RUN_TEACHER" -eq 1 ]; then
    CMD="$CMD --run_teacher"
fi
if [ "$RUN_BASELINE" -eq 1 ]; then
    CMD="$CMD --run_baseline"
fi
if [ "$RUN_STUDENT" -eq 1 ]; then
    CMD="$CMD --run_student"
fi
if [ "$USE_KD" -eq 1 ]; then
    CMD="$CMD --use_kd"
fi

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
if [ -n "$STUDENT_CKPT" ]; then
    CMD="$CMD --student_checkpoint $STUDENT_CKPT"
fi
if [ "$SKIP_SAVE_MODELS" -eq 1 ]; then
    CMD="$CMD --skip_save_models"
fi

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
