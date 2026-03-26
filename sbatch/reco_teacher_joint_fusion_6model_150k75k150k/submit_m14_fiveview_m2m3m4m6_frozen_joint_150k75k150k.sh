#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M14="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m14_fiveview_m2m3m4m6_frozen_joint_150k75k150k.sh"

if [[ ! -f "$S_M14" ]]; then
  echo "Missing script: $S_M14" >&2
  exit 1
fi

jid=$(sbatch "$S_M14" | awk '{print $4}')
echo "Submitted m14 five-view frozen+joint run: ${jid}"
