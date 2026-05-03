#!/usr/bin/env bash
set -euo pipefail

# Submit one follow-up sbatch job that waits for the six m38 prefix jobs.
#
# Usage:
#   bash sbatch/reco_teacher_joint_fusion_6model_150k75k150k/submit_after_m38_prefix6_dependency.sh <runner_script>
#
# Example:
#   bash sbatch/reco_teacher_joint_fusion_6model_150k75k150k/submit_after_m38_prefix6_dependency.sh \
#     sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint31_bin_gated_fusion_valsel.sh
#
# Optional override:
#   DEPENDENCY_JOB_IDS="id1 id2 id3 id4 id5 id6" bash .../submit_after_m38_prefix6_dependency.sh <runner_script>

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <runner_script>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER_SCRIPT="$1"
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
  echo "Missing runner script: ${RUNNER_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${DEPENDENCY_JOB_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  JOB_IDS=( ${DEPENDENCY_JOB_IDS} )
else
  JOB_IDS=(21253956 21253957 21253958 21253959 21253960 21253961)
fi

if [[ "${#JOB_IDS[@]}" -lt 1 ]]; then
  echo "No dependency job IDs provided." >&2
  exit 1
fi

DEPS="$(IFS=:; echo "${JOB_IDS[*]}")"
OUT="$(sbatch --dependency="afterok:${DEPS}" "${RUNNER_SCRIPT}")"
JOBID="$(echo "${OUT}" | awk '{print $4}')"

echo "Submitted dependent job: ${JOBID}"
echo "  runner     : ${RUNNER_SCRIPT}"
echo "  dependency : afterok:${DEPS}"

