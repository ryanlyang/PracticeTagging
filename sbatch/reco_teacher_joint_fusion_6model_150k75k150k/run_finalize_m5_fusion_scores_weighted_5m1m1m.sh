#!/usr/bin/env bash
#SBATCH --job-name=m5fin
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/finalize_m5_fusion_scores_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/finalize_m5_fusion_scores_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE="${BASE:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
RUN_NAME="${RUN_NAME:-model5_joint_s01_full_weighted_5m1m1m_seed0}"
SAVE_DIR="${SAVE_DIR:-${BASE}/model5_joint_s01_full_weighted_5m1m1m}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:--1}"
TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONHASHSEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
for f in \
  "${RUN_DIR}/teacher.pt" \
  "${RUN_DIR}/baseline.pt" \
  "${RUN_DIR}/offline_reconstructor_stage2.pt" \
  "${RUN_DIR}/dual_joint_stage2.pt" \
  "${RUN_DIR}/offline_reconstructor_stageC_selected_pre_eval.pt" \
  "${RUN_DIR}/dual_joint_stageC_selected_pre_eval.pt"
do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing required input: ${f}" >&2
    exit 1
  fi
done

CMD=(
  python -u finalize_m5_fusion_scores_from_checkpoints.py
  --train_path "${TRAIN_PATH}"
  --use_train_weights
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --added_target_scale 0.90
  --report_target_tpr 0.50
)

echo "============================================================"
echo "Finalize m5 fusion scores from saved checkpoints"
echo "Run dir: ${RUN_DIR}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${RUN_DIR}"
