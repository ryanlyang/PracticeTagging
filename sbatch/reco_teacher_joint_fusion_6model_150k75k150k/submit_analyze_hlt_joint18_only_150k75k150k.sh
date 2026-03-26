#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found in PATH. Run this on the SLURM submit host." >&2
  exit 1
fi

S_AN="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint18_fusion_150k75k150k.sh"
if [[ ! -f "$S_AN" ]]; then
  echo "Missing script: $S_AN" >&2
  exit 1
fi

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
M2_RUN_DIR="${M2_RUN_DIR:-${BASE_DIR}/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M3_RUN_DIR="${M3_RUN_DIR:-${BASE_DIR}/model3_recoteacher_s09/model3_recoteacher_s09_150k75k150k_seed0}"

# Colon-separated job IDs. If empty, submits analysis immediately.
EXISTING_DEPS="${EXISTING_DEPS:-}"

if [[ -n "${EXISTING_DEPS}" ]]; then
  j_an=$(
    sbatch \
      --dependency="afterok:${EXISTING_DEPS}" \
      --export=ALL,M2_RUN_DIR="${M2_RUN_DIR}",M3_RUN_DIR="${M3_RUN_DIR}" \
      "$S_AN" | awk '{print $4}'
  )
  echo "Submitted analysis-only 18-model job: ${j_an}"
  echo "  dependency=afterok:${EXISTING_DEPS}"
else
  j_an=$(
    sbatch \
      --export=ALL,M2_RUN_DIR="${M2_RUN_DIR}",M3_RUN_DIR="${M3_RUN_DIR}" \
      "$S_AN" | awk '{print $4}'
  )
  echo "Submitted analysis-only 18-model job immediately: ${j_an}"
fi

echo "  using M2_RUN_DIR=${M2_RUN_DIR}"
echo "  using M3_RUN_DIR=${M3_RUN_DIR}"
