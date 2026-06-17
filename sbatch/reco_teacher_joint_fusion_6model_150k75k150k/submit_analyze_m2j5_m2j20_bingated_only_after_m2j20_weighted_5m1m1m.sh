#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"
RUN_ANALYZE="${ROOT}/run_analyze_m2j5_m2j20_bingated_only_weighted_5m1m1m.sh"
M2J20_JOB_ID="${M2J20_JOB_ID:-21344786}"

job_id=$(
  sbatch \
    --parsable \
    --chdir="${PWD}" \
    --dependency="afterok:${M2J20_JOB_ID}" \
    "${RUN_ANALYZE}"
)

echo "Queued M2J5/M2J20 bin-gated-only analyze job: ${job_id}"
echo "Dependency: afterok:${M2J20_JOB_ID}"
