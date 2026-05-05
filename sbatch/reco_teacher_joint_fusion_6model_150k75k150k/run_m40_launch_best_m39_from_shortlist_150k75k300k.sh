#!/usr/bin/env bash
#SBATCH --job-name=m40m39ln
#SBATCH --partition=tier3
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m40_launch_best_m39_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m40_launch_best_m39_%j.err

set -euo pipefail

SWEEP_SAVE_DIR="${SWEEP_SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/m40_constituent_codebook}"
SWEEP_RUN_NAME="${SWEEP_RUN_NAME:-m40_quant_sweep_150k75k300k_seed0}"
SHORTLIST_PATH="${SHORTLIST_PATH:-${SWEEP_SAVE_DIR}/${SWEEP_RUN_NAME}/shortlist.json}"
M39_SUBMIT_SCRIPT="${M39_SUBMIT_SCRIPT:-sbatch/reco_teacher_joint_fusion_6model_150k75k150k/submit_m39_prefixspecialist_detresid_multicand_150k75k300k_after6_stage2keep_sweep.sh}"

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

if [[ ! -f "${SHORTLIST_PATH}" ]]; then
  echo "ERROR: shortlist not found: ${SHORTLIST_PATH}" >&2
  exit 1
fi

read -r CODEBOOK_PATH CODEBOOK_LABEL < <(
python - <<'PY' "${SHORTLIST_PATH}"
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    arr = json.load(f)
if not isinstance(arr, list) or len(arr) == 0:
    raise SystemExit("shortlist is empty")
best = arr[0]
cb = str(best.get("codebook_dir", "")).strip()
lb = str(best.get("label", "best")).strip()
if not cb:
    raise SystemExit("best shortlist entry has no codebook_dir")
print(cb, lb)
PY
)

if [[ -z "${CODEBOOK_PATH}" ]]; then
  echo "ERROR: parsed empty CODEBOOK_PATH from shortlist." >&2
  exit 1
fi

label_safe="$(echo "${CODEBOOK_LABEL}" | tr -cs '[:alnum:]_-' '_')"
BASE_RUN_NAME="model39_prefixspecialist_detresid_multicand_150k75k300k_${label_safe}_seed0"
STAGE2_RUN_NAME="model39_prefix6_stage2_150k75k300k_${label_safe}_seed0"
STAGE2_RUN_PREFIX="model39_prefix6_stage2_${label_safe}_keepm"

export CODEBOOK_PATH
export CODEBOOK_LABEL
export BASE_RUN_NAME
export STAGE2_RUN_NAME
export STAGE2_RUN_PREFIX

echo "Using shortlist: ${SHORTLIST_PATH}"
echo "Best codebook: ${CODEBOOK_PATH} (${CODEBOOK_LABEL})"
echo "Submitting m39 with BASE_RUN_NAME=${BASE_RUN_NAME}"

bash "${M39_SUBMIT_SCRIPT}"
