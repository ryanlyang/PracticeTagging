#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M7="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m7_stageA_residual_hlt_150k75k150k.sh"

if [[ ! -f "$S_M7" ]]; then
  echo "Missing script: $S_M7" >&2
  exit 1
fi

jid=$(sbatch "$S_M7" | awk '{print $4}')
echo "Submitted Model-7 StageA residual run: ${jid}"
