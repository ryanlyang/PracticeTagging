#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m38_k6_seeded_m28_detresid_multicand_carryboost_150k75k300k.sh"
BASE_RUN_NAME="model38_k6_seeded_m28_detresid_multicand_carryboost_150k75k300k_seed0"
PREFIXES=(0 3 6 9 12 15)

for pfx in "${PREFIXES[@]}"; do
  sbatch \
    --job-name="m38cbp${pfx}" \
    --output="offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m38_k6_seeded_m28_detresid_multicand_carryboost_p${pfx}_%j.out" \
    --error="offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m38_k6_seeded_m28_detresid_multicand_carryboost_p${pfx}_%j.err" \
    --export=ALL,RUN_NAME="${BASE_RUN_NAME}_pfx${pfx}",SEED_CANDIDATE_K=1,SEED_KEEP_M=1,SEED_MAX_PREFIX="${pfx}" \
    "${RUNNER}"
done
