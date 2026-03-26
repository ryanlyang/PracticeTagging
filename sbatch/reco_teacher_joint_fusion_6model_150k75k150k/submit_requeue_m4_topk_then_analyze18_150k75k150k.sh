#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found in PATH. Run this on the SLURM submit host." >&2
  exit 1
fi

S40="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_40max_150k75k150k.sh"
S60="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_60max_150k75k150k.sh"
S80="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_80max_150k75k150k.sh"
SAN="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint18_fusion_150k75k150k.sh"

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
M2_RUN_DIR="${M2_RUN_DIR:-${BASE_DIR}/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M3_RUN_DIR="${M3_RUN_DIR:-${BASE_DIR}/model3_recoteacher_s09/model3_recoteacher_s09_150k75k150k_seed0}"

for s in "$S40" "$S60" "$S80" "$SAN"; do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

# Existing jobs to wait on (override via env if needed).
# Default excludes m2/m3 so analysis uses existing older m2/m3 checkpoints.
EXISTING_DEPS="${EXISTING_DEPS:-21118441:21118442:21118443:21118444:21118451:21118452:21118727}"

j40=$(sbatch "$S40" | awk '{print $4}')
j60=$(sbatch "$S60" | awk '{print $4}')
j80=$(sbatch "$S80" | awk '{print $4}')

if [[ -n "${EXISTING_DEPS}" ]]; then
  deps="${EXISTING_DEPS}:${j40}:${j60}:${j80}"
else
  deps="${j40}:${j60}:${j80}"
fi

jan=$(
  sbatch \
    --dependency="afterok:${deps}" \
    --export=ALL,M2_RUN_DIR="${M2_RUN_DIR}",M3_RUN_DIR="${M3_RUN_DIR}" \
    "$SAN" | awk '{print $4}'
)

echo "Submitted requeued m4-topk jobs:"
echo "  k40 = ${j40}"
echo "  k60 = ${j60}"
echo "  k80 = ${j80}"
echo "Submitted analysis job:"
echo "  an18f = ${jan}"
echo "  using existing m2 run dir: ${M2_RUN_DIR}"
echo "  using existing m3 run dir: ${M3_RUN_DIR}"
echo "  dependency=afterok:${deps}"
