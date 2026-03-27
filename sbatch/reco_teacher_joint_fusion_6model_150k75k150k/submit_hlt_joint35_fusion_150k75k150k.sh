#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
JOB_ID=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint35_fusion_150k75k150k.sh | awk '{print $4}')
echo "Submitted 31-model fusion analysis: ${JOB_ID}"
