#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M10="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m10_recoteacher_s01_corrected_antioverlap_150k75k150k.sh"

if [[ ! -f "$S_M10" ]]; then
  echo "Missing script: $S_M10" >&2
  exit 1
fi

jid=$(sbatch "$S_M10" | awk '{print $4}')
echo "Submitted Model-10 anti-overlap m4-style run: ${jid}"
