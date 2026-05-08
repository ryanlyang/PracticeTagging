#!/usr/bin/env bash
#SBATCH --job-name=jt12fst
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/train_toptag_offline_fused_student_joint12_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/train_toptag_offline_fused_student_joint12_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-toptag_offline_fused_student_joint12_weighted_150k150k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/offline_fused_student_joint12_weighted_150k150k300k}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k/fused_targets_train_val_test.npz}"
TARGET_KEY="${TARGET_KEY:-probs_fused_overall}"
TARGET_SPLIT_SCHEME="${TARGET_SPLIT_SCHEME:-train_val_test}"
TARGET_SOURCE_SPLITS_NPZ="${TARGET_SOURCE_SPLITS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"

N_TRAIN_JETS="${N_TRAIN_JETS:-600000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-150000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"

mkdir -p "${SAVE_DIR}"

CMD=(
  python train_toptag_offline_fused_student.py
  --train_path "${TRAIN_PATH}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --batch_size 512
  --epochs 80
  --patience 15
  --warmup_epochs 5
  --lr 3e-4
  --weight_decay 1e-5
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --use_train_weights
  --fused_targets_npz "${FUSED_TARGETS_NPZ}"
  --target_key "${TARGET_KEY}"
  --target_split_scheme "${TARGET_SPLIT_SCHEME}"
  --target_source_splits_npz "${TARGET_SOURCE_SPLITS_NPZ}"
)

echo "============================================================"
echo "Train top-tagging offline fused student"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Targets: ${FUSED_TARGETS_NPZ}"
echo "Source splits: ${TARGET_SOURCE_SPLITS_NPZ}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
