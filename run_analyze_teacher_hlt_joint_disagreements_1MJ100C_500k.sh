#!/usr/bin/env bash
# Extended disagreement analysis (Teacher vs HLT vs Joint) on a fresh 500k slice.
#
# Submit:
#   sbatch run_analyze_teacher_hlt_joint_disagreements_1MJ100C_500k.sh

#SBATCH --job-name=discTHJ500k
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=offline_reconstructor_logs/disagreement_thj_1MJ100C_500k_%j.out
#SBATCH --error=offline_reconstructor_logs/disagreement_thj_1MJ100C_500k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_1MJ100C}"
SAVE_SUBDIR="${SAVE_SUBDIR:-teacher_hlt_joint_disagreement_analysis_500k}"
N_EVAL_JETS="${N_EVAL_JETS:-500000}"
OFFSET_EVAL_JETS="${OFFSET_EVAL_JETS:--1}"   # -1 => start after original training range
TARGET_TPR="${TARGET_TPR:-0.50}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-test}"  # test or val
THRESHOLD_VAL_FRAC="${THRESHOLD_VAL_FRAC:-0.20}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:--1}"
BUCKET_MIN_COUNT="${BUCKET_MIN_COUNT:-1000}"
BUCKET_MIN_POS="${BUCKET_MIN_POS:-300}"
BUCKET_MIN_NEG="${BUCKET_MIN_NEG:-300}"
MAX_EXPORT_PER_SUBSET="${MAX_EXPORT_PER_SUBSET:-100000}"
EXPORT_ALL_SUBSET_JETS="${EXPORT_ALL_SUBSET_JETS:-0}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
DUAL_CKPT_NAME="${DUAL_CKPT_NAME:-dual_joint.pt}"
RECO_CKPT_NAME="${RECO_CKPT_NAME:-offline_reconstructor.pt}"
DATA_FILE="${DATA_FILE:-}"  # optional override, e.g. /path/to/test.h5

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"
ANALYZE_SCRIPT="${ANALYZE_SCRIPT:-${SUBMIT_DIR}/analyze_teacher_hlt_joint_disagreements.py}"

if [[ ! -f "${ANALYZE_SCRIPT}" ]]; then
  echo "ERROR: analyzer script not found: ${ANALYZE_SCRIPT}" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cmd=(
  python "${ANALYZE_SCRIPT}"
  --run_dir "${RUN_DIR}"
  --save_subdir "${SAVE_SUBDIR}"
  --n_eval_jets "${N_EVAL_JETS}"
  --offset_eval_jets "${OFFSET_EVAL_JETS}"
  --target_tpr "${TARGET_TPR}"
  --threshold_source "${THRESHOLD_SOURCE}"
  --threshold_val_frac "${THRESHOLD_VAL_FRAC}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --bucket_min_count "${BUCKET_MIN_COUNT}"
  --bucket_min_pos "${BUCKET_MIN_POS}"
  --bucket_min_neg "${BUCKET_MIN_NEG}"
  --max_export_per_subset "${MAX_EXPORT_PER_SUBSET}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --dual_ckpt_name "${DUAL_CKPT_NAME}"
  --reco_ckpt_name "${RECO_CKPT_NAME}"
)

if [[ -n "${DATA_FILE}" ]]; then
  cmd+=(--data_file "${DATA_FILE}")
fi
if [[ "${EXPORT_ALL_SUBSET_JETS}" -eq 1 ]]; then
  cmd+=(--export_all_subset_jets)
fi

echo "Running extended Teacher/HLT/Joint disagreement analysis:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"
