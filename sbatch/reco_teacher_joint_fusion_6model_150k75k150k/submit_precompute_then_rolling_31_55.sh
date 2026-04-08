#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PRECOMP_SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_precompute_hlt_joint55_devpool_scores_1m.sh"
RUN31_SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint31_rolling_chunk_greedy_precomputed_1m.sh"
RUN55_SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint55_rolling_chunk_greedy_precomputed_1m.sh"

if [[ ! -f "${PRECOMP_SCRIPT}" ]]; then
  echo "Missing script: ${PRECOMP_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${RUN31_SCRIPT}" ]]; then
  echo "Missing script: ${RUN31_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${RUN55_SCRIPT}" ]]; then
  echo "Missing script: ${RUN55_SCRIPT}" >&2
  exit 1
fi

PRE_OUT="$(sbatch "${PRECOMP_SCRIPT}")"
PRE_JOBID="$(echo "${PRE_OUT}" | awk '{print $4}')"
if [[ -z "${PRE_JOBID}" ]]; then
  echo "Failed to parse precompute job id from: ${PRE_OUT}" >&2
  exit 1
fi

OUT31="$(sbatch --dependency=afterok:${PRE_JOBID} "${RUN31_SCRIPT}")"
JOB31="$(echo "${OUT31}" | awk '{print $4}')"
OUT55="$(sbatch --dependency=afterok:${PRE_JOBID} "${RUN55_SCRIPT}")"
JOB55="$(echo "${OUT55}" | awk '{print $4}')"

echo "Submitted jobs:"
echo "  precompute: ${PRE_JOBID}"
echo "  analyze31:  ${JOB31} (afterok:${PRE_JOBID})"
echo "  analyze55:  ${JOB55} (afterok:${PRE_JOBID})"

