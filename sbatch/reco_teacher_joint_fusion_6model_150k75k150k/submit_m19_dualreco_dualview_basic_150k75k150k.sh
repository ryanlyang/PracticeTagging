#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
JOB_ID=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m19_dualreco_dualview_basic_150k75k150k.sh | awk '{print $4}')
echo "Submitted m19 basic dual-reco dualview: ${JOB_ID}"
