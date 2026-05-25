#!/usr/bin/env bash
set -euo pipefail

# Requeue helper for the JetClass 12-run m2-hybrid + dependent stacked fusion.
# Uses a fresh save root by default so existing completed runs are preserved.

PARTITION="${PARTITION:-tier3}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
TRAIN_MEM="${TRAIN_MEM:-64G}"
FUSION_MEM="${FUSION_MEM:-96G}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"

RERUN_TAG="${RERUN_TAG:-rerun_$(date +%Y%m%d_%H%M%S)}"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-checkpoints/jetclass_joint_dualview_reruns}"
SAVE_DIR="${SAVE_DIR:-${BASE_SAVE_ROOT}/${RERUN_TAG}}"
OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/twelve_model_250k_m2hybrid_stacked_acc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "Requeue JetClass 12-run m2-hybrid pipeline"
echo "Tag:        ${RERUN_TAG}"
echo "Partition:  ${PARTITION}"
echo "Time limit: ${TIME_LIMIT}"
echo "Train mem:  ${TRAIN_MEM}"
echo "Fusion mem: ${FUSION_MEM}"
echo "Save dir:   ${SAVE_DIR}"
echo "Out dir:    ${OUT_DIR}"
echo "============================================================"

PARTITION="${PARTITION}" \
TIME_LIMIT="${TIME_LIMIT}" \
TRAIN_MEM="${TRAIN_MEM}" \
FUSION_MEM="${FUSION_MEM}" \
SAVE_DIR="${SAVE_DIR}" \
OUT_DIR="${OUT_DIR}" \
OPTIMIZE_FOR="${OPTIMIZE_FOR}" \
bash submit_jetclass_m2hybrid_joint12_250k_pipeline.sh
