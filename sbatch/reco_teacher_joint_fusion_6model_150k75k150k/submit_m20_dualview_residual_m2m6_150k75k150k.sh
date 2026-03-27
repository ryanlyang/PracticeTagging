#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
JOB_ID=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m20_dualview_residual_m2m6_150k75k150k.sh | awk '{print $4}')
echo "Submitted m20 dualview residual (m2+m6): ${JOB_ID}"
