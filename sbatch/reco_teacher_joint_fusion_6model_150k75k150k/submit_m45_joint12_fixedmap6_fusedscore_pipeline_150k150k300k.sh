#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_BUILD="${RUN_BUILD:-${SCRIPT_DIR}/run_build_joint12_fused_targets_weighted_150k150k300k.sh}"
RUN_FINAL="${RUN_FINAL:-${SCRIPT_DIR}/run_m45_joint12_fixedmap6_fusedscore_stagea_teacher_weighted_150k150k300k.sh}"

SCORES_NPZ="${SCORES_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_scores.npz}"
REPORT_JSON="${REPORT_JSON:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_report.json}"
FIXED_MODELS="${FIXED_MODELS:-joint_delta,dual_m17_antioverlap,offdrop_mid,joint_s01,dual_m15_offdrop_high,offdrop_high}"
FIXED_PREFIX="${FIXED_PREFIX:-probs_fixedmap}"
FIXED_REDUCTION="${FIXED_REDUCTION:-mean}"

TARGETS_DIR="${TARGETS_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_fixedmap6_weighted_150k150k300k}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-model45_joint12_fixedmap6_fusedscore_stagea_teacher_weighted_150k150k300k_seed0}"

UPSTREAM_JOB_IDS="${UPSTREAM_JOB_IDS:-}"

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
    --export=ALL,SCORES_NPZ="${SCORES_NPZ}",OUT_DIR="${TARGETS_DIR}",REPORT_JSON="${REPORT_JSON}",FIXED_MODELS="${FIXED_MODELS}",FIXED_PREFIX="${FIXED_PREFIX}",FIXED_REDUCTION="${FIXED_REDUCTION}" \
    "${RUN_BUILD}")
else
  echo "Submitting fixed-map build job"
  job_build=$(sbatch --parsable \
    --export=ALL,SCORES_NPZ="${SCORES_NPZ}",OUT_DIR="${TARGETS_DIR}",REPORT_JSON="${REPORT_JSON}",FIXED_MODELS="${FIXED_MODELS}",FIXED_PREFIX="${FIXED_PREFIX}",FIXED_REDUCTION="${FIXED_REDUCTION}" \
    "${RUN_BUILD}")
fi
echo "  build: ${job_build}"

job_final=$(sbatch --parsable \
  --dependency=afterok:${job_build} \
  --export=ALL,STAGEA_FUSED_TARGETS_NPZ="${TARGETS_DIR}/fused_targets_train_val_test.npz",STAGEA_FUSED_TARGETS_KEY="${FIXED_PREFIX}",RUN_NAME="${FINAL_RUN_NAME}" \
  "${RUN_FINAL}")
echo "  final: ${job_final}"

echo "============================================================"
echo "Queued Model-45 fixed-map6 direct fused-score pipeline"
echo "build jobid: ${job_build}"
echo "final jobid: ${job_final}"
echo "============================================================"
