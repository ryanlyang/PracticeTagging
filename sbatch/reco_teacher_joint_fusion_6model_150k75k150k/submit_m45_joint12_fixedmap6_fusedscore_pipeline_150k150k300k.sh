#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_BUILD="${RUN_BUILD:-${SCRIPT_DIR}/run_build_joint12_fused_targets_weighted_150k150k300k.sh}"
RUN_FINAL="${RUN_FINAL:-${SCRIPT_DIR}/run_m45_joint12_fixedmap6_fusedscore_stagea_teacher_weighted_150k150k300k.sh}"

SCORES_NPZ="${SCORES_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_scores.npz}"
REPORT_JSON="${REPORT_JSON:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_report.json}"
DEFAULT_FIXED_MODELS="joint_delta,dual_m17_antioverlap,offdrop_mid,joint_s01,dual_m15_offdrop_high,offdrop_high"
# Intentionally use FIXED_MODELS_OVERRIDE (not FIXED_MODELS) so stale shell vars do not silently change the mapping.
FIXED_MODELS="${FIXED_MODELS_OVERRIDE:-${DEFAULT_FIXED_MODELS}}"
FIXED_PREFIX="${FIXED_PREFIX:-probs_fixedmap}"
FIXED_REDUCTION="${FIXED_REDUCTION:-mean}"
FIXED_STRATEGY="${FIXED_STRATEGY:-greedy_global}"
FIXED_ANCHOR_MODEL="${FIXED_ANCHOR_MODEL:-joint_delta}"
FIXED_TARGET_TPRS="${FIXED_TARGET_TPRS:-0.50,0.30}"
FIXED_TPR_REDUCTION="${FIXED_TPR_REDUCTION:-mean}"
FIXED_CALIBRATION="${FIXED_CALIBRATION:-iso}"
FIXED_W_STEP="${FIXED_W_STEP:-0.005}"
FIXED_MAX_ADD="${FIXED_MAX_ADD:-6}"
FIXED_MIN_IMPROVE="${FIXED_MIN_IMPROVE:-2e-7}"
FIXED_HEAD_SELECT_MODE="${FIXED_HEAD_SELECT_MODE:-best_val_fpr}"
FIXED_HEAD_SELECT_TPR="${FIXED_HEAD_SELECT_TPR:-0.50}"

TARGETS_DIR="${TARGETS_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_fixedmap6_strong_weighted_150k150k300k}"
FINAL_SAVE_DIR="${FINAL_SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model45_joint12_fixedmap6_strong_fusedscore_stagea_teacher_weighted_150k150k300k}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-model45_joint12_fixedmap6_strong_fusedscore_stagea_teacher_weighted_150k150k300k_seed0}"

UPSTREAM_JOB_IDS="${UPSTREAM_JOB_IDS:-}"

FIXED_MODELS="${FIXED_MODELS// /}"
IFS=',' read -r -a _raw_fixed_models <<<"${FIXED_MODELS}"
_fixed_models=()
for m in "${_raw_fixed_models[@]}"; do
  if [[ -n "${m}" ]]; then
    _fixed_models+=("${m}")
  fi
done
if [[ ${#_fixed_models[@]} -ne 6 ]]; then
  echo "ERROR: fixed-map6 expects exactly 6 models, got ${#_fixed_models[@]}." >&2
  echo "Provide FIXED_MODELS_OVERRIDE with 6 comma-separated models." >&2
  exit 2
fi
FIXED_MODELS="$(IFS=,; echo "${_fixed_models[*]}")"

for f in "${RUN_BUILD}" "${RUN_FINAL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing runner: ${f}" >&2
    exit 1
  fi
done

if [[ -n "${UPSTREAM_JOB_IDS}" ]]; then
  echo "Submitting fixed-map build job with dependency afterok:${UPSTREAM_JOB_IDS}"
  job_build=$(sbatch --parsable \
    --dependency=afterok:${UPSTREAM_JOB_IDS} \
    --export=ALL,SCORES_NPZ="${SCORES_NPZ}",OUT_DIR="${TARGETS_DIR}",REPORT_JSON="${REPORT_JSON}",FIXED_MODELS="${FIXED_MODELS}",FIXED_PREFIX="${FIXED_PREFIX}",FIXED_REDUCTION="${FIXED_REDUCTION}",FIXED_STRATEGY="${FIXED_STRATEGY}",FIXED_ANCHOR_MODEL="${FIXED_ANCHOR_MODEL}",FIXED_TARGET_TPRS="${FIXED_TARGET_TPRS}",FIXED_TPR_REDUCTION="${FIXED_TPR_REDUCTION}",FIXED_CALIBRATION="${FIXED_CALIBRATION}",FIXED_W_STEP="${FIXED_W_STEP}",FIXED_MAX_ADD="${FIXED_MAX_ADD}",FIXED_MIN_IMPROVE="${FIXED_MIN_IMPROVE}",FIXED_HEAD_SELECT_MODE="${FIXED_HEAD_SELECT_MODE}",FIXED_HEAD_SELECT_TPR="${FIXED_HEAD_SELECT_TPR}" \
    "${RUN_BUILD}")
else
  echo "Submitting fixed-map build job"
  job_build=$(sbatch --parsable \
    --export=ALL,SCORES_NPZ="${SCORES_NPZ}",OUT_DIR="${TARGETS_DIR}",REPORT_JSON="${REPORT_JSON}",FIXED_MODELS="${FIXED_MODELS}",FIXED_PREFIX="${FIXED_PREFIX}",FIXED_REDUCTION="${FIXED_REDUCTION}",FIXED_STRATEGY="${FIXED_STRATEGY}",FIXED_ANCHOR_MODEL="${FIXED_ANCHOR_MODEL}",FIXED_TARGET_TPRS="${FIXED_TARGET_TPRS}",FIXED_TPR_REDUCTION="${FIXED_TPR_REDUCTION}",FIXED_CALIBRATION="${FIXED_CALIBRATION}",FIXED_W_STEP="${FIXED_W_STEP}",FIXED_MAX_ADD="${FIXED_MAX_ADD}",FIXED_MIN_IMPROVE="${FIXED_MIN_IMPROVE}",FIXED_HEAD_SELECT_MODE="${FIXED_HEAD_SELECT_MODE}",FIXED_HEAD_SELECT_TPR="${FIXED_HEAD_SELECT_TPR}" \
    "${RUN_BUILD}")
fi
echo "  build: ${job_build}"

job_final=$(sbatch --parsable \
  --dependency=afterok:${job_build} \
  --export=ALL,STAGEA_FUSED_TARGETS_NPZ="${TARGETS_DIR}/fused_targets_train_val_test.npz",STAGEA_FUSED_TARGETS_KEY="${FIXED_PREFIX}",SAVE_DIR="${FINAL_SAVE_DIR}",RUN_NAME="${FINAL_RUN_NAME}" \
  "${RUN_FINAL}")
echo "  final: ${job_final}"

echo "============================================================"
echo "Queued Model-45 fixed-map6 direct fused-score pipeline"
echo "fixed models: ${FIXED_MODELS}"
echo "fixed strategy: ${FIXED_STRATEGY} (anchor=${FIXED_ANCHOR_MODEL}, tprs=${FIXED_TARGET_TPRS}, cal=${FIXED_CALIBRATION}, w_step=${FIXED_W_STEP})"
echo "build jobid: ${job_build}"
echo "final jobid: ${job_final}"
echo "============================================================"
