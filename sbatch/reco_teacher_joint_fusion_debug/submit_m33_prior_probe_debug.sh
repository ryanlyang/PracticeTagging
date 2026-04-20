#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S="sbatch/reco_teacher_joint_fusion_debug/run_m33_prior_probe_debug.sh"
if [[ ! -f "$S" ]]; then
  echo "Missing script: $S" >&2
  exit 1
fi

jid=$(sbatch "$S" | awk '{print $4}')
echo "Submitted m33 prior probe debug: ${jid}"
