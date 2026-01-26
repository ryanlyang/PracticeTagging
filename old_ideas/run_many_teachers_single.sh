#!/bin/bash

#SBATCH --job-name=many_teachers
#SBATCH --output=many_teachers_logs/many_teachers_%j.out
#SBATCH --error=many_teachers_logs/many_teachers_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1

mkdir -p many_teachers_logs

echo "=========================================="
echo "Many Teachers (Half-HLT -> 3/2 HLT)"
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

RUN_NAME=${RUN_NAME:-"half_teachers_quick"}
SAVE_DIR=${SAVE_DIR:-"checkpoints/many_teachers"}
TRAIN_PATH=${TRAIN_PATH:-""}
N_TRAIN_JETS=${N_TRAIN_JETS:-30000}
MAX_CONSTITS=${MAX_CONSTITS:-80}
SKIP_SAVE_MODELS=${SKIP_SAVE_MODELS:-0}

N_TEACHERS=${N_TEACHERS:-3}
SINGLE_TEACHER_IDX=${SINGLE_TEACHER_IDX:-0}
HALF_HLT_STRENGTH=${HALF_HLT_STRENGTH:-0.5}
STUDENT_HLT_STRENGTH=${STUDENT_HLT_STRENGTH:-1.5}
HALF_HLT_SEEDS=${HALF_HLT_SEEDS:-""}
HALF_HLT_SEED_BASE=${HALF_HLT_SEED_BASE:-123}
HALF_HLT_SEED_STEP=${HALF_HLT_SEED_STEP:-333}
STUDENT_HLT_SEED=${STUDENT_HLT_SEED:-999}

TEMP_INIT=${TEMP_INIT:-7.0}
TEMP_FINAL=${TEMP_FINAL:-""}
ALPHA_INIT=${ALPHA_INIT:-0.5}
ALPHA_FINAL=${ALPHA_FINAL:-""}

echo "Hyperparameters:"
echo "  Run Name: $RUN_NAME"
echo "  n_train_jets: $N_TRAIN_JETS"
echo "  n_teachers: $N_TEACHERS (single idx=$SINGLE_TEACHER_IDX)"
echo "  half_hlt_strength: $HALF_HLT_STRENGTH"
echo "  student_hlt_strength: $STUDENT_HLT_STRENGTH"
echo ""

CMD="python many_teachers.py \
    --mode half_teachers \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --n_train_jets $N_TRAIN_JETS \
    --max_constits $MAX_CONSTITS \
    --n_teachers $N_TEACHERS \
    --single_teacher_idx $SINGLE_TEACHER_IDX \
    --half_hlt_strength $HALF_HLT_STRENGTH \
    --student_hlt_strength $STUDENT_HLT_STRENGTH \
    --half_hlt_seed_base $HALF_HLT_SEED_BASE \
    --half_hlt_seed_step $HALF_HLT_SEED_STEP \
    --student_hlt_seed $STUDENT_HLT_SEED \
    --temp_init $TEMP_INIT \
    --alpha_init $ALPHA_INIT"

if [ -n "$HALF_HLT_SEEDS" ]; then
    CMD="$CMD --half_hlt_seeds $HALF_HLT_SEEDS"
fi

if [ -n "$TEMP_FINAL" ]; then
    CMD="$CMD --temp_final $TEMP_FINAL"
fi

if [ -n "$ALPHA_FINAL" ]; then
    CMD="$CMD --alpha_final $ALPHA_FINAL"
fi

if [ -n "$TRAIN_PATH" ]; then
    CMD="$CMD --train_path $TRAIN_PATH"
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
