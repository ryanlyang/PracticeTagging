#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

RUN_MID_RECOONLY="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_recoonly.sh"
RUN_HIGH_RECOONLY="${ROOT}/run_m15_dualreco_dualview_offdrop_high_weighted_5m1m1m_recoonly.sh"
RUN_MID_TAGGER="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_tagger_from_reco.sh"
RUN_HIGH_TAGGER="${ROOT}/run_m15_dualreco_dualview_offdrop_high_weighted_5m1m1m_tagger_from_reco.sh"
RUN_ANALYZE="${ROOT}/run_analyze_hlt_joint12_bin_gated_fusion_valsel_weighted_5m1m1m.sh"

QUEUE_ANALYZE="${QUEUE_ANALYZE:-0}"
OTHER_DEP_IDS="${OTHER_DEP_IDS:-}"

j_mid_reco=$(sbatch "${RUN_MID_RECOONLY}" | awk '{print $4}')
j_high_reco=$(sbatch "${RUN_HIGH_RECOONLY}" | awk '{print $4}')

j_mid_tag=$(sbatch --dependency="afterok:${j_mid_reco}" "${RUN_MID_TAGGER}" | awk '{print $4}')
j_high_tag=$(sbatch --dependency="afterok:${j_high_reco}" "${RUN_HIGH_TAGGER}" | awk '{print $4}')

echo "Submitted M15 split chain jobs:"
echo "  mid_recoonly : ${j_mid_reco}"
echo "  high_recoonly: ${j_high_reco}"
echo "  mid_tagger   : ${j_mid_tag} (afterok:${j_mid_reco})"
echo "  high_tagger  : ${j_high_tag} (afterok:${j_high_reco})"

if [[ "${QUEUE_ANALYZE}" == "1" ]]; then
  dep="${j_mid_tag}:${j_high_tag}"
  if [[ -n "${OTHER_DEP_IDS}" ]]; then
    dep="${OTHER_DEP_IDS}:${dep}"
  fi
  j_an=$(sbatch --dependency="afterok:${dep}" "${RUN_ANALYZE}" | awk '{print $4}')
  echo "Queued analyze job: ${j_an}"
  echo "Analyze dependency: afterok:${dep}"
else
  echo "Analyze not queued (QUEUE_ANALYZE=${QUEUE_ANALYZE})."
  echo "Set QUEUE_ANALYZE=1 and optionally OTHER_DEP_IDS=\"id1:id2:...\" to queue it."
fi
