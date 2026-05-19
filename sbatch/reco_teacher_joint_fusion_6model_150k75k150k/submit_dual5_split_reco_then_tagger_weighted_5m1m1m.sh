#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

RUN_M12_RECOONLY="${ROOT}/run_m12_dualreco_dualview_feat_noscale_weighted_5m1m1m_recoonly.sh"
RUN_M12_TAGGER="${ROOT}/run_m12_dualreco_dualview_feat_noscale_weighted_5m1m1m_tagger_from_reco.sh"

RUN_M15MID_RECOONLY="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_recoonly.sh"
RUN_M15MID_TAGGER="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_tagger_from_reco.sh"
RUN_M15HIGH_RECOONLY="${ROOT}/run_m15_dualreco_dualview_offdrop_high_weighted_5m1m1m_recoonly.sh"
RUN_M15HIGH_TAGGER="${ROOT}/run_m15_dualreco_dualview_offdrop_high_weighted_5m1m1m_tagger_from_reco.sh"

RUN_M16_RECOONLY="${ROOT}/run_m16_dualreco_dualview_topk60_weighted_5m1m1m_recoonly.sh"
RUN_M16_TAGGER="${ROOT}/run_m16_dualreco_dualview_topk60_weighted_5m1m1m_tagger_from_reco.sh"

RUN_M17_RECOONLY="${ROOT}/run_m17_dualreco_dualview_antioverlap_weighted_5m1m1m_recoonly.sh"
RUN_M17_TAGGER="${ROOT}/run_m17_dualreco_dualview_antioverlap_weighted_5m1m1m_tagger_from_reco.sh"

RUN_ANALYZE="${ROOT}/run_analyze_hlt_joint12_bin_gated_fusion_valsel_weighted_5m1m1m.sh"

QUEUE_ANALYZE="${QUEUE_ANALYZE:-0}"
OTHER_DEP_IDS="${OTHER_DEP_IDS:-}"
INCLUDE_HLT_CANDIDATE="${INCLUDE_HLT_CANDIDATE:-1}"
STEP1_REF_NPZ="${STEP1_REF_NPZ:-}"

M12_RUN_DIR="${M12_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_from_recoonly/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_from_recoonly}"
M15MID_RUN_DIR="${M15MID_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_from_recoonly/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_from_recoonly}"
M15HIGH_RUN_DIR="${M15HIGH_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_from_recoonly/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_from_recoonly}"
M16_RUN_DIR="${M16_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model16_dualreco_dualview_topk60_weighted_5m1m1m_from_recoonly/model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_from_recoonly}"
M17_RUN_DIR="${M17_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_recoonly/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_recoonly}"

j12_reco=$(sbatch "${RUN_M12_RECOONLY}" | awk '{print $4}')
j15m_reco=$(sbatch "${RUN_M15MID_RECOONLY}" | awk '{print $4}')
j15h_reco=$(sbatch "${RUN_M15HIGH_RECOONLY}" | awk '{print $4}')
j16_reco=$(sbatch "${RUN_M16_RECOONLY}" | awk '{print $4}')
j17_reco=$(sbatch "${RUN_M17_RECOONLY}" | awk '{print $4}')

j12_tag=$(sbatch --dependency="afterok:${j12_reco}" "${RUN_M12_TAGGER}" | awk '{print $4}')
j15m_tag=$(sbatch --dependency="afterok:${j15m_reco}" "${RUN_M15MID_TAGGER}" | awk '{print $4}')
j15h_tag=$(sbatch --dependency="afterok:${j15h_reco}" "${RUN_M15HIGH_TAGGER}" | awk '{print $4}')
j16_tag=$(sbatch --dependency="afterok:${j16_reco}" "${RUN_M16_TAGGER}" | awk '{print $4}')
j17_tag=$(sbatch --dependency="afterok:${j17_reco}" "${RUN_M17_TAGGER}" | awk '{print $4}')

echo "Submitted dual-reco split chain jobs:"
echo "  m12 reco/tag : ${j12_reco} -> ${j12_tag}"
echo "  m15m reco/tag: ${j15m_reco} -> ${j15m_tag}"
echo "  m15h reco/tag: ${j15h_reco} -> ${j15h_tag}"
echo "  m16 reco/tag : ${j16_reco} -> ${j16_tag}"
echo "  m17 reco/tag : ${j17_reco} -> ${j17_tag}"

if [[ "${QUEUE_ANALYZE}" == "1" ]]; then
  dep="${j12_tag}:${j15m_tag}:${j15h_tag}:${j16_tag}:${j17_tag}"
  if [[ -n "${OTHER_DEP_IDS}" ]]; then
    dep="${OTHER_DEP_IDS}:${dep}"
  fi

  an_export="ALL,M12_RUN_DIR=${M12_RUN_DIR},M15MID_RUN_DIR=${M15MID_RUN_DIR},M15HIGH_RUN_DIR=${M15HIGH_RUN_DIR},M16_RUN_DIR=${M16_RUN_DIR},M17_RUN_DIR=${M17_RUN_DIR},INCLUDE_HLT_CANDIDATE=${INCLUDE_HLT_CANDIDATE}"
  if [[ -n "${STEP1_REF_NPZ}" ]]; then
    an_export="${an_export},STEP1_REF_NPZ=${STEP1_REF_NPZ}"
  fi

  j_an=$(sbatch --dependency="afterok:${dep}" --export="${an_export}" "${RUN_ANALYZE}" | awk '{print $4}')
  echo "Queued analyze job: ${j_an}"
  echo "Analyze dependency: afterok:${dep}"
else
  echo "Analyze not queued (QUEUE_ANALYZE=${QUEUE_ANALYZE})."
  echo "Set QUEUE_ANALYZE=1 and optionally OTHER_DEP_IDS=\"id1:id2:...\"."
fi
