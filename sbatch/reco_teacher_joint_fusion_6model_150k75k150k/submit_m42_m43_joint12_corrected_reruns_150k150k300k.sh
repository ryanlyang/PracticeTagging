#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN42="${HERE}/run_m42_joint12_fusedscore_stagea_teacher_weighted_150k150k300k.sh"
RUN43="${HERE}/run_m43_joint12_fusedadv_stagea_dualkd_weighted_150k150k300k.sh"

dep="${DEP_JOB_ID:-}"

if [[ -n "${dep}" ]]; then
  jid42=$(sbatch --dependency="afterok:${dep}" --parsable "${RUN42}")
  jid43=$(sbatch --dependency="afterok:${dep}" --parsable "${RUN43}")
else
  jid42=$(sbatch --parsable "${RUN42}")
  jid43=$(sbatch --parsable "${RUN43}")
fi

echo "Submitted corrected reruns:"
echo "  m42=${jid42}"
echo "  m43=${jid43}"
