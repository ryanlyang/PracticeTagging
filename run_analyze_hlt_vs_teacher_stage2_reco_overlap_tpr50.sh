#!/usr/bin/env bash
# Compute HLT vs teacher-on-Stage2-reco overlap/correlation at TPR=50.
#
# Submit:
#   sbatch run_analyze_hlt_vs_teacher_stage2_reco_overlap_tpr50.sh

#SBATCH --job-name=ovlS2TH50
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=offline_reconstructor_logs/overlap_hlt_teacher_stage2/ovl_hlt_teacher_s2_%j.out
#SBATCH --error=offline_reconstructor_logs/overlap_hlt_teacher_stage2/ovl_hlt_teacher_s2_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/overlap_hlt_teacher_stage2}"
mkdir -p "${LOG_DIR}"

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd_rho090_75k35k200k_100c_noflags_seed0}"
TARGET_TPR="${TARGET_TPR:-0.50}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
RECO_BATCH_SIZE="${RECO_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WEIGHT_THRESHOLD="${WEIGHT_THRESHOLD:-0.03}"
DISABLE_BUDGET_TOPK="${DISABLE_BUDGET_TOPK:-0}"
DATA_FILE="${DATA_FILE:-}"

TEACHER_CKPT="${TEACHER_CKPT:-teacher.pt}"
HLT_CKPT="${HLT_CKPT:-baseline.pt}"
RECO_CKPT="${RECO_CKPT:-offline_reconstructor_stage2.pt}"
OUTPUT_NAME="${OUTPUT_NAME:-hlt_vs_teacher_on_stage2_reco_overlap_tpr50.json}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

SCRIPT_PATH="${SCRIPT_PATH:-${SUBMIT_DIR}/analyze_hlt_vs_teacher_stage2_reco_overlap.py}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: analysis script not found: ${SCRIPT_PATH}" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cmd=(
  python "${SCRIPT_PATH}"
  --run_dir "${RUN_DIR}"
  --target_tpr "${TARGET_TPR}"
  --device "${DEVICE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --reco_batch_size "${RECO_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --weight_threshold "${WEIGHT_THRESHOLD}"
  --teacher_ckpt "${TEACHER_CKPT}"
  --hlt_ckpt "${HLT_CKPT}"
  --reco_ckpt "${RECO_CKPT}"
  --output_name "${OUTPUT_NAME}"
)

if [[ -n "${DATA_FILE}" ]]; then
  cmd+=(--data_file "${DATA_FILE}")
fi
if [[ "${DISABLE_BUDGET_TOPK}" -eq 1 ]]; then
  cmd+=(--disable_budget_topk)
fi

echo "============================================================"
echo "HLT vs Teacher(Stage2 Reco) overlap analysis"
echo "Run dir       : ${RUN_DIR}"
echo "Target TPR    : ${TARGET_TPR}"
echo "Device        : ${DEVICE}"
echo "Output JSON   : ${RUN_DIR}/${OUTPUT_NAME}"
echo "============================================================"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"


echo
echo "Done."
echo "Output: ${RUN_DIR}/${OUTPUT_NAME}"
