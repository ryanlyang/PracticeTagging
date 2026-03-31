#!/usr/bin/env bash
#SBATCH --job-name=m2diagpc10
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_diag_ptclamp10_constraint_saturation_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_diag_ptclamp10_constraint_saturation_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

DEFAULT_RUN_SUBPATH="reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_fulltrain_prog_unfreeze_ptclamp10/model2_joint_delta005_fulltrain_prog_unfreeze_ptclamp10_150k75k150k_seed0"
DEFAULT_RUN_DIR_CHECKPOINTS="checkpoints/${DEFAULT_RUN_SUBPATH}"
DEFAULT_RUN_DIR_DOWNLOAD="download_checkpoints/${DEFAULT_RUN_SUBPATH}"

if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -d "${DEFAULT_RUN_DIR_CHECKPOINTS}" ]]; then
    RUN_DIR="${DEFAULT_RUN_DIR_CHECKPOINTS}"
  elif [[ -d "${DEFAULT_RUN_DIR_DOWNLOAD}" ]]; then
    RUN_DIR="${DEFAULT_RUN_DIR_DOWNLOAD}"
  else
    RUN_DIR="${DEFAULT_RUN_DIR_CHECKPOINTS}"
  fi
fi

OUT_DIR="${OUT_DIR:-${RUN_DIR}/constraint_saturation}"

STAGE2_RECO_CKPT="${STAGE2_RECO_CKPT:-${RUN_DIR}/offline_reconstructor_stage2.pt}"
JOINT_RECO_CKPT="${JOINT_RECO_CKPT:-${RUN_DIR}/offline_reconstructor.pt}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_BATCHES="${MAX_BATCHES:--1}"
DEVICE="${DEVICE:-cuda}"
STAGE_SCALE="${STAGE_SCALE:-1.0}"

# Match diagnostic threshold to this ablation's token log-pT cap.
DIAG_LOGPT_MIN="${DIAG_LOGPT_MIN:--9.0}"
DIAG_LOGPT_MAX="${DIAG_LOGPT_MAX:-10.0}"
DIAG_LOGE_MIN="${DIAG_LOGE_MIN:--9.0}"
DIAG_LOGE_MAX="${DIAG_LOGE_MAX:-11.0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUT_DIR}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${STAGE2_RECO_CKPT}" ]]; then
  echo "ERROR: STAGE2_RECO_CKPT missing: ${STAGE2_RECO_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${JOINT_RECO_CKPT}" ]]; then
  echo "ERROR: JOINT_RECO_CKPT missing: ${JOINT_RECO_CKPT}" >&2
  exit 1
fi

echo "============================================================"
echo "Constraint Saturation Diagnostics (PtClamp10 Ablation)"
echo "Run dir:        ${RUN_DIR}"
echo "Split:          ${SPLIT}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Device:         ${DEVICE}"
echo "Stage scale:    ${STAGE_SCALE}"
echo "diag_logpt_min: ${DIAG_LOGPT_MIN}"
echo "diag_logpt_max: ${DIAG_LOGPT_MAX}"
echo "diag_loge_min:  ${DIAG_LOGE_MIN}"
echo "diag_loge_max:  ${DIAG_LOGE_MAX}"
echo "Stage2 ckpt:    ${STAGE2_RECO_CKPT}"
echo "Joint ckpt:     ${JOINT_RECO_CKPT}"
echo "Output dir:     ${OUT_DIR}"
echo "============================================================"

COMMON_ARGS=(
  --run_dir "${RUN_DIR}"
  --split "${SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_batches "${MAX_BATCHES}"
  --device "${DEVICE}"
  --stage_scale "${STAGE_SCALE}"
  --diag_logpt_min "${DIAG_LOGPT_MIN}"
  --diag_logpt_max "${DIAG_LOGPT_MAX}"
  --diag_loge_min "${DIAG_LOGE_MIN}"
  --diag_loge_max "${DIAG_LOGE_MAX}"
  --reco_module offline_reconstructor_no_gt_local30kv2_ptclamp10
  --reco_class OfflineReconstructorPtClamp10
)

CMD_STAGE2=(
  python analyze_reco_constraint_saturation.py
  "${COMMON_ARGS[@]}"
  --checkpoint "${STAGE2_RECO_CKPT}"
  --report_json "${OUT_DIR}/stage2_auc_constraint_saturation_${SPLIT}.json"
)

CMD_JOINT=(
  python analyze_reco_constraint_saturation.py
  "${COMMON_ARGS[@]}"
  --checkpoint "${JOINT_RECO_CKPT}"
  --report_json "${OUT_DIR}/joint_auc_constraint_saturation_${SPLIT}.json"
)

echo
echo "--- Stage2 AUC Reconstructor Diagnostic ---"
printf ' %q' "${CMD_STAGE2[@]}"
echo
"${CMD_STAGE2[@]}"

echo
echo "--- Joint AUC Reconstructor Diagnostic ---"
printf ' %q' "${CMD_JOINT[@]}"
echo
"${CMD_JOINT[@]}"

echo
echo "Done. Reports:"
echo "  ${OUT_DIR}/stage2_auc_constraint_saturation_${SPLIT}.json"
echo "  ${OUT_DIR}/joint_auc_constraint_saturation_${SPLIT}.json"

