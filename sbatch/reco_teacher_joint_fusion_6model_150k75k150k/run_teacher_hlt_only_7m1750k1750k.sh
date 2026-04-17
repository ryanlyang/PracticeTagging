#!/usr/bin/env bash
#SBATCH --job-name=th7m175
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=263G
#SBATCH --time=4-12:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_7m1750k1750k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_7m1750k1750k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-teacher_hlt_only_7m1750k1750k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_7m1750k1750k/teacher_hlt_only}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
TEST_PATH="${TEST_PATH:-./data/test.h5}"
TEST_OFFSET_JETS="${TEST_OFFSET_JETS:-0}"

N_TRAIN_JETS="${N_TRAIN_JETS:-10500000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-7000000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-1750000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-1750000}"
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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
  --train_path "${TRAIN_PATH}"
  --test_path "${TEST_PATH}"
  --test_offset_jets "${TEST_OFFSET_JETS}"
  --use_train_weights
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --selection_metric auc
  --step1_only
  --device "${DEVICE}"
)

echo "============================================================"
echo "Teacher + HLT Baseline Only (7M/1.75M/1.75M split, real train/test H5, weighted)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Train path: ${TRAIN_PATH}"
echo "Test path:  ${TEST_PATH}"
echo "Data: n_train_jets=${N_TRAIN_JETS}, split=${N_TRAIN_SPLIT}/${N_VAL_SPLIT}/${N_TEST_SPLIT}, offset=${OFFSET_JETS}, test_offset=${TEST_OFFSET_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
