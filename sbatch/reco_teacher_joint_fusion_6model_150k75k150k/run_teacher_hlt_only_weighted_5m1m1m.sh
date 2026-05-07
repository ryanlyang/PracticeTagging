#!/usr/bin/env bash
#SBATCH --job-name=th5m1m
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=320G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-teacher_hlt_only_weighted_5m1m1m_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_weighted_5m1m1m}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
FORCE_M5_STEP1="${FORCE_M5_STEP1:-1}"
STEP1_RESUME_FROM_CHECKPOINTS="${STEP1_RESUME_FROM_CHECKPOINTS:-1}"
STEP1_CKPT_EVERY_EPOCHS="${STEP1_CKPT_EVERY_EPOCHS:-1}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
TEST_PATH="${TEST_PATH:-./data/test.h5}"
TEST_OFFSET_JETS="${TEST_OFFSET_JETS:-0}"

N_TRAIN_JETS="${N_TRAIN_JETS:-7000000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-5000000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-1000000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-1000000}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --step1_checkpoint_every_epochs "${STEP1_CKPT_EVERY_EPOCHS}"
  --device "${DEVICE}"
)

if [[ "${FORCE_M5_STEP1}" == "1" ]]; then
  CMD+=(--force_m5_step1)
fi
if [[ "${STEP1_RESUME_FROM_CHECKPOINTS}" == "1" ]]; then
  CMD+=(--step1_resume_from_checkpoints)
fi

echo "============================================================"
echo "Teacher + HLT Baseline Only (5M/1M/1M, weighted)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Train path: ${TRAIN_PATH}"
echo "Test path:  ${TEST_PATH}"
echo "Data: n_train_jets=${N_TRAIN_JETS}, split=${N_TRAIN_SPLIT}/${N_VAL_SPLIT}/${N_TEST_SPLIT}, offset=${OFFSET_JETS}, test_offset=${TEST_OFFSET_JETS}"
echo "Step1 ckpt/resume: every=${STEP1_CKPT_EVERY_EPOCHS}, resume=${STEP1_RESUME_FROM_CHECKPOINTS}, force_m5=${FORCE_M5_STEP1}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
