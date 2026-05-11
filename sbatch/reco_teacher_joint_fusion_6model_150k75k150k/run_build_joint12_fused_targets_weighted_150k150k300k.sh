#!/usr/bin/env bash
#SBATCH --job-name=jt12fuse
#SBATCH --partition=tier3
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/build_joint12_fused_targets_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/build_joint12_fused_targets_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

SCORES_NPZ="${SCORES_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_scores.npz}"
OUT_DIR="${OUT_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k}"
OVERALL_FAMILY="${OVERALL_FAMILY:-bin}"
REDUCTION="${REDUCTION:-mean}"
REPORT_JSON="${REPORT_JSON:-}"
FIXED_MODELS="${FIXED_MODELS:-}"
FIXED_PREFIX="${FIXED_PREFIX:-probs_fixedmap}"
FIXED_REDUCTION="${FIXED_REDUCTION:-mean}"
FIXED_STRATEGY="${FIXED_STRATEGY:-mean}"
FIXED_ANCHOR_MODEL="${FIXED_ANCHOR_MODEL:-}"
FIXED_TARGET_TPRS="${FIXED_TARGET_TPRS:-0.50,0.30}"
FIXED_TPR_REDUCTION="${FIXED_TPR_REDUCTION:-mean}"
FIXED_CALIBRATION="${FIXED_CALIBRATION:-iso}"
FIXED_W_STEP="${FIXED_W_STEP:-0.005}"
FIXED_MAX_ADD="${FIXED_MAX_ADD:-6}"
FIXED_MIN_IMPROVE="${FIXED_MIN_IMPROVE:-2e-7}"
FIXED_HEAD_SELECT_MODE="${FIXED_HEAD_SELECT_MODE:-best_val_fpr}"
FIXED_HEAD_SELECT_TPR="${FIXED_HEAD_SELECT_TPR:-0.50}"
FIXED_INCLUDE_PER_MODEL="${FIXED_INCLUDE_PER_MODEL:-0}"
ALLOW_FIT_REF_OVERLAP="${ALLOW_FIT_REF_OVERLAP:-0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CMD=(
  python build_toptag_fused_targets_from_joint12_bingated.py
  --scores_npz "${SCORES_NPZ}"
  --out_dir "${OUT_DIR}"
  --overall_family "${OVERALL_FAMILY}"
  --reduction "${REDUCTION}"
)

if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi
if [[ -n "${FIXED_MODELS}" ]]; then
  CMD+=(
    --fixed_models "${FIXED_MODELS}"
    --fixed_prefix "${FIXED_PREFIX}"
    --fixed_reduction "${FIXED_REDUCTION}"
    --fixed_strategy "${FIXED_STRATEGY}"
    --fixed_target_tprs "${FIXED_TARGET_TPRS}"
    --fixed_tpr_reduction "${FIXED_TPR_REDUCTION}"
    --fixed_calibration "${FIXED_CALIBRATION}"
    --fixed_w_step "${FIXED_W_STEP}"
    --fixed_max_add "${FIXED_MAX_ADD}"
    --fixed_min_improve "${FIXED_MIN_IMPROVE}"
    --fixed_head_select_mode "${FIXED_HEAD_SELECT_MODE}"
    --fixed_head_select_tpr "${FIXED_HEAD_SELECT_TPR}"
  )
  if [[ -n "${FIXED_ANCHOR_MODEL}" ]]; then
    CMD+=(--fixed_anchor_model "${FIXED_ANCHOR_MODEL}")
  fi
  if [[ "${FIXED_INCLUDE_PER_MODEL}" == "1" ]]; then
    CMD+=(--fixed_include_per_model)
  fi
fi

if [[ "${ALLOW_FIT_REF_OVERLAP}" == "1" ]]; then
  CMD+=(--allow_fit_ref_overlap)
fi

echo "============================================================"
echo "Build joint12 fused targets (top-tagging)"
echo "Scores: ${SCORES_NPZ}"
echo "Out:    ${OUT_DIR}"
if [[ -n "${FIXED_MODELS}" ]]; then
  echo "Fixed-map models: ${FIXED_MODELS}"
  echo "Fixed-map prefix: ${FIXED_PREFIX}"
  echo "Fixed-map strategy: ${FIXED_STRATEGY} (reduction=${FIXED_REDUCTION}, tprs=${FIXED_TARGET_TPRS}, tpr_reduction=${FIXED_TPR_REDUCTION})"
  echo "Fixed-map calibration: ${FIXED_CALIBRATION}, w_step=${FIXED_W_STEP}, max_add=${FIXED_MAX_ADD}, min_improve=${FIXED_MIN_IMPROVE}"
  echo "Fixed-map head select: mode=${FIXED_HEAD_SELECT_MODE}, tpr=${FIXED_HEAD_SELECT_TPR}, anchor=${FIXED_ANCHOR_MODEL:-<first_model>}, per_model=${FIXED_INCLUDE_PER_MODEL}"
fi
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${OUT_DIR}"
