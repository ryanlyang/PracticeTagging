#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_AN="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint5_fusion_150k75k150k.sh"

if [[ ! -f "$S_AN" ]]; then
  echo "Missing script: $S_AN" >&2
  exit 1
fi

jid=$(sbatch "$S_AN" | awk '{print $4}')
echo "Submitted five-model analysis: ${jid}"
