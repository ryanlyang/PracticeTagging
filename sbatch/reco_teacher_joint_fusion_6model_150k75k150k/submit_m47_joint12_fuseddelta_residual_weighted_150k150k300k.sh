#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${HERE}/run_m47_joint12_fuseddelta_residual_weighted_150k150k300k.sh"

if [[ -n "${DEP_JOB_ID:-}" ]]; then
  jid=$(sbatch --dependency="afterok:${DEP_JOB_ID}" --parsable "${RUNNER}")
  echo "Submitted m47 with dependency afterok:${DEP_JOB_ID}: ${jid}"
else
  jid=$(sbatch --parsable "${RUNNER}")
  echo "Submitted m47: ${jid}"
fi
