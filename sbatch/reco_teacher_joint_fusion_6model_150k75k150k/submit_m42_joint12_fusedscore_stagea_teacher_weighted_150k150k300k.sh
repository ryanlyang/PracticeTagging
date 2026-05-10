#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${HERE}/run_m42_joint12_fusedscore_stagea_teacher_weighted_150k150k300k.sh"

if [[ -n "${DEP_JOB_ID:-}" ]]; then
  jid=$(sbatch --dependency="afterok:${DEP_JOB_ID}" --parsable "${RUNNER}")
  echo "Submitted m42 with dependency afterok:${DEP_JOB_ID}: ${jid}"
else
  jid=$(sbatch --parsable "${RUNNER}")
  echo "Submitted m42: ${jid}"
fi
