#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_BUILD="${RUN_BUILD:-${SCRIPT_DIR}/run_build_joint12_fused_targets_weighted_150k150k300k.sh}"
RUN_STUDENT="${RUN_STUDENT:-${SCRIPT_DIR}/run_train_toptag_offline_fused_student_joint12_weighted_150k150k300k.sh}"
RUN_FINAL="${RUN_FINAL:-${SCRIPT_DIR}/run_m41_joint12_fusedstudent_stagea_teacher_weighted_150k150k300k.sh}"

SCORES_NPZ="${SCORES_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_scores.npz}"
TARGETS_DIR="${TARGETS_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k}"
STUDENT_SAVE_DIR="${STUDENT_SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/offline_fused_student_joint12_weighted_150k150k300k}"
STUDENT_RUN_NAME="${STUDENT_RUN_NAME:-toptag_offline_fused_student_joint12_weighted_150k150k300k_seed0}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-model41_joint12_fusedstudent_stagea_teacher_weighted_150k150k300k_seed0}"
TARGET_SOURCE_SPLITS_NPZ="${TARGET_SOURCE_SPLITS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"

UPSTREAM_JOB_IDS="${UPSTREAM_JOB_IDS:-}"

for f in "${RUN_BUILD}" "${RUN_STUDENT}" "${RUN_FINAL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing runner: ${f}" >&2
    exit 1
  fi
done

if [[ -n "${UPSTREAM_JOB_IDS}" ]]; then
  echo "Submitting build job with dependency afterok:${UPSTREAM_JOB_IDS}"
  job_build=$(sbatch --parsable \
    --dependency=afterok:${UPSTREAM_JOB_IDS} \
    --export=ALL,SCORES_NPZ="${SCORES_NPZ}",OUT_DIR="${TARGETS_DIR}" \
    "${RUN_BUILD}")
else
  echo "Submitting build job"
  job_build=$(sbatch --parsable \
    --export=ALL,SCORES_NPZ="${SCORES_NPZ}",OUT_DIR="${TARGETS_DIR}" \
    "${RUN_BUILD}")
fi
echo "  build:   ${job_build}"

job_student=$(sbatch --parsable \
  --dependency=afterok:${job_build} \
  --export=ALL,FUSED_TARGETS_NPZ="${TARGETS_DIR}/fused_targets_train_val_test.npz",TARGET_SOURCE_SPLITS_NPZ="${TARGET_SOURCE_SPLITS_NPZ}",SAVE_DIR="${STUDENT_SAVE_DIR}",RUN_NAME="${STUDENT_RUN_NAME}" \
  "${RUN_STUDENT}")
echo "  student: ${job_student}"

job_final=$(sbatch --parsable \
  --dependency=afterok:${job_student} \
  --export=ALL,STAGEA_TEACHER_CKPT="${STUDENT_SAVE_DIR}/${STUDENT_RUN_NAME}/teacher.pt",RUN_NAME="${FINAL_RUN_NAME}" \
  "${RUN_FINAL}")
echo "  final:   ${job_final}"

echo "============================================================"
echo "Queued Strategy-2 pipeline (fused-student Stage-A teacher)"
echo "build   jobid: ${job_build}"
echo "student jobid: ${job_student}"
echo "final   jobid: ${job_final}"
echo "============================================================"
