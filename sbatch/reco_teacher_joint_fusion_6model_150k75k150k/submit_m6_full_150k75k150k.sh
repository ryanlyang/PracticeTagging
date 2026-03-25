#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M6="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m6_concat_stagea_corrected_full_150k75k150k.sh"

if [[ ! -f "$S_M6" ]]; then
  echo "Missing script: $S_M6" >&2
  exit 1
fi

jid=$(sbatch "$S_M6" | awk '{print $4}')
echo "Submitted full model6 concat run: ${jid}"
