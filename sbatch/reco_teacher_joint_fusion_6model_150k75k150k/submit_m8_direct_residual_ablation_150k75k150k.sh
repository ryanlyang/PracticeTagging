#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M8="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m8_direct_residual_ablation_150k75k150k.sh"

if [[ ! -f "$S_M8" ]]; then
  echo "Missing script: $S_M8" >&2
  exit 1
fi

jid=$(sbatch "$S_M8" | awk '{print $4}')
echo "Submitted Model-8 direct-residual ablation: ${jid}"
