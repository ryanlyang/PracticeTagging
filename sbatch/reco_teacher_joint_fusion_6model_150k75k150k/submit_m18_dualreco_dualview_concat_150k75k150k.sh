#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
JOB_ID=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m18_dualreco_dualview_concat_150k75k150k.sh | awk '{print $4}')
echo "Submitted m18 concat dual-reco dualview: ${JOB_ID}"
