#!/usr/bin/env bash
set -euo pipefail

# Submit m38 prefix6 stage2 after six prefix jobs succeed.
#
# Usage:
#   bash sbatch/reco_teacher_joint_fusion_6model_150k75k150k/submit_m38_prefix6_stage2_after_current6.sh
#
# Optional override:
#   DEPENDENCY_JOB_IDS="id1 id2 id3 id4 id5 id6" bash .../submit_m38_prefix6_stage2_after_current6.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m38_prefix6_stage2_150k75k300k.sh"
if [[ ! -f "${RUNNER}" ]]; then
  echo "Missing runner: ${RUNNER}" >&2
  exit 1
fi

if [[ -n "${DEPENDENCY_JOB_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  JOB_IDS=( ${DEPENDENCY_JOB_IDS} )
else
  JOB_IDS=(21253956 21253957 21253958 21253959 21253960 21253961)
fi

DEPS="$(IFS=:; echo "${JOB_IDS[*]}")"
OUT="$(sbatch --dependency="afterok:${DEPS}" "${RUNNER}")"
JOBID="$(echo "${OUT}" | awk '{print $4}')"

echo "Submitted stage2 job: ${JOBID}"
echo "  runner     : ${RUNNER}"
echo "  dependency : afterok:${DEPS}"

