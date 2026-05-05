#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"
RUNNER="${ROOT}/run_m40_quantization_sweep_150k75k300k.sh"

jid=$(sbatch "${RUNNER}" | awk '{print $4}')
echo "Submitted m40 quantization sweep: ${jid}"
