#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"
SWEEP_RUNNER="${ROOT}/run_m40_quantization_sweep_150k75k300k.sh"
LAUNCH_RUNNER="${ROOT}/run_m40_launch_best_m39_from_shortlist_150k75k300k.sh"

out1=$(sbatch "${SWEEP_RUNNER}")
jid1=$(echo "${out1}" | awk '{print $4}')
echo "Submitted m40 sweep: ${jid1}"

out2=$(sbatch --dependency="afterok:${jid1}" "${LAUNCH_RUNNER}")
jid2=$(echo "${out2}" | awk '{print $4}')
echo "Submitted dependent launch-best-m39 job: ${jid2}"
echo "Dependency: afterok:${jid1}"
