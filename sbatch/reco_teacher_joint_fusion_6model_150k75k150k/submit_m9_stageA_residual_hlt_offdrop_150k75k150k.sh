#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M9="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m9_stageA_residual_hlt_offdrop_150k75k150k.sh"

if [[ ! -f "$S_M9" ]]; then
  echo "Missing script: $S_M9" >&2
  exit 1
fi

jid=$(sbatch "$S_M9" | awk '{print $4}')
echo "Submitted Model-9 StageA offdrop residual run: ${jid}"
