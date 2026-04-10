#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m2_joint_hungarian_iterrefine_effsplit_150k75k150k.sh"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing script: ${SCRIPT}" >&2
  exit 1
fi

OUT_FIXED_COLD="$(sbatch --export=ALL,HLT_EDIT_MODE=fixed,WARM_START_MODE=cold "${SCRIPT}")"
JOB_FIXED_COLD="$(echo "${OUT_FIXED_COLD}" | awk '{print $4}')"

OUT_TINY_COLD="$(sbatch --export=ALL,HLT_EDIT_MODE=tiny_edit,WARM_START_MODE=cold "${SCRIPT}")"
JOB_TINY_COLD="$(echo "${OUT_TINY_COLD}" | awk '{print $4}')"

OUT_FIXED_WARM="$(sbatch --export=ALL,HLT_EDIT_MODE=fixed,WARM_START_MODE=warm "${SCRIPT}")"
JOB_FIXED_WARM="$(echo "${OUT_FIXED_WARM}" | awk '{print $4}')"

OUT_TINY_WARM="$(sbatch --export=ALL,HLT_EDIT_MODE=tiny_edit,WARM_START_MODE=warm "${SCRIPT}")"
JOB_TINY_WARM="$(echo "${OUT_TINY_WARM}" | awk '{print $4}')"

echo "Submitted:"
echo "  fixed+cold:      ${JOB_FIXED_COLD}"
echo "  tiny_edit+cold:  ${JOB_TINY_COLD}"
echo "  fixed+warm:      ${JOB_FIXED_WARM}"
echo "  tiny_edit+warm:  ${JOB_TINY_WARM}"
