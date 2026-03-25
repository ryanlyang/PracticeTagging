#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S40="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_40max_150k75k150k.sh"
S60="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_60max_150k75k150k.sh"
S80="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_80max_150k75k150k.sh"

for s in "$S40" "$S60" "$S80"; do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

j40=$(sbatch "$S40" | awk '{print $4}')
j60=$(sbatch "$S60" | awk '{print $4}')
j80=$(sbatch "$S80" | awk '{print $4}')

echo "Submitted m4-topk jobs:"
echo "  k40: ${j40}"
echo "  k60: ${j60}"
echo "  k80: ${j80}"
