#!/usr/bin/env bash
set -euo pipefail

# Orchestration:
# 1) Submit two teacher runs.
# 2) Submit bin-gated fusion analysis after both succeed.
# 3) Submit fused-two-teacher KD run after fusion succeeds.

RUNNER_A="${RUNNER_A:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual.sh}"
RUNNER_B="${RUNNER_B:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_two_model_bin_gated_valsel_v1hltplus25_pair.sh}"
RUNNER_KD="${RUNNER_KD:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_fusedkd_two_teacher.sh}"

SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_A_NAME="${RUN_A_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
RUN_B_NAME="${RUN_B_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25_gentok56}"
FUSION_OUT_DIR="${FUSION_OUT_DIR:-${SAVE_DIR}/fusion_reports/${RUN_A_NAME}__AND__${RUN_B_NAME}__bin_gated_valsel}"
KD_RUN_NAME="${KD_RUN_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_v1hltplus25_fusedkd_two_teacher_from_${RUN_A_NAME}_and_${RUN_B_NAME}}"

if [[ ! -f "${RUNNER_A}" ]]; then
  echo "Missing runner A: ${RUNNER_A}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER_B}" ]]; then
  echo "Missing runner B: ${RUNNER_B}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER_FUSION}" ]]; then
  echo "Missing fusion runner: ${RUNNER_FUSION}" >&2
  exit 1
fi
if [[ ! -f "${RUNNER_KD}" ]]; then
  echo "Missing KD runner: ${RUNNER_KD}" >&2
  exit 1
fi

RUN_A_DIR="${SAVE_DIR}/${RUN_A_NAME}"
RUN_B_DIR="${SAVE_DIR}/${RUN_B_NAME}"

echo "Submitting teacher run A:"
echo "  ${RUNNER_A}"
echo "  RUN_NAME=${RUN_A_NAME}"
job_a=$(sbatch --parsable --export=ALL,RUN_NAME="${RUN_A_NAME}" "${RUNNER_A}")
echo "  Job A ID: ${job_a}"

echo "Submitting teacher run B:"
echo "  ${RUNNER_B}"
echo "  RUN_NAME=${RUN_B_NAME}"
job_b=$(sbatch --parsable --export=ALL,RUN_NAME="${RUN_B_NAME}" "${RUNNER_B}")
echo "  Job B ID: ${job_b}"

echo "Submitting dependent two-model fusion analysis (afterok:${job_a}:${job_b})"
job_fuse=$(sbatch --parsable \
  --dependency=afterok:${job_a}:${job_b} \
  --export=ALL,RUN_A_DIR="${RUN_A_DIR}",RUN_B_DIR="${RUN_B_DIR}",OUT_DIR="${FUSION_OUT_DIR}" \
  "${RUNNER_FUSION}")
echo "  Fusion job ID: ${job_fuse}"

echo "Submitting dependent fused-KD run (afterok:${job_fuse})"
job_kd=$(sbatch --parsable \
  --dependency=afterok:${job_fuse} \
  --export=ALL,RUN_NAME="${KD_RUN_NAME}",TEACHER_A_RUN="${RUN_A_DIR}",TEACHER_B_RUN="${RUN_B_DIR}" \
  "${RUNNER_KD}")
echo "  KD job ID: ${job_kd}"

echo "============================================================"
echo "Queued chain complete"
echo "A train job:   ${job_a}"
echo "B train job:   ${job_b}"
echo "Fusion job:    ${job_fuse}"
echo "Fused-KD job:  ${job_kd}"
echo "Teacher A dir: ${RUN_A_DIR}"
echo "Teacher B dir: ${RUN_B_DIR}"
echo "Fusion outdir: ${FUSION_OUT_DIR}"
echo "KD run name:   ${KD_RUN_NAME}"
echo "============================================================"

