#!/usr/bin/env bash
set -euo pipefail

# Control experiment:
#   Train five HLT-only baselines on one fixed split/HLT view, using different
#   model-training seeds only, then run the same stacked-logistic fusion.

PARTITION="${PARTITION:-tier3}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-4-00:00:00}"
FUSION_TIME_LIMIT="${FUSION_TIME_LIMIT:-2-00:00:00}"
TRAIN_MEM="${TRAIN_MEM:-256G}"
FUSION_MEM="${FUSION_MEM:-160G}"

RUNNER_TRAIN="${RUNNER_TRAIN:-run_train_jetclass_hlt5_seed_ensemble_1m250k1m_fixedhlt.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_hlt5_seed_ensemble_stacked_fusion_1m250k1m_fixedhlt.sh}"

for f in "${RUNNER_TRAIN}" "${RUNNER_FUSION}"; do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

submit_train() {
  sbatch --parsable \
    --job-name="jcHLT5" \
    --partition="${PARTITION}" \
    --time="${TRAIN_TIME_LIMIT}" \
    --mem="${TRAIN_MEM}" \
    "${RUNNER_TRAIN}"
}

submit_fusion() {
  local dep="$1"
  sbatch --parsable \
    --job-name="jcHLT5F" \
    --partition="${PARTITION}" \
    --time="${FUSION_TIME_LIMIT}" \
    --mem="${FUSION_MEM}" \
    --dependency="afterok:${dep}" \
    "${RUNNER_FUSION}"
}

echo "Submitting JetClass HLT-only 5-seed ensemble control"
echo "Partition: ${PARTITION}"
echo "Train time: ${TRAIN_TIME_LIMIT}"
echo "Fusion time: ${FUSION_TIME_LIMIT}"

jtrain=$(submit_train)
echo "  TRAIN  ${jtrain} jcHLT5"

jfuse=$(submit_fusion "${jtrain}")
echo "  FUSION ${jfuse} jcHLT5F"

echo "============================================================"
echo "Queued HLT-only 5-seed ensemble control"
echo "Train job:  ${jtrain}"
echo "Fusion job: ${jfuse}"
echo "Fusion dependency: afterok:${jtrain}"
echo "Fusion out: checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/hlt5_1m250k1m_fixedhlt_stacked_acc"
echo "============================================================"

