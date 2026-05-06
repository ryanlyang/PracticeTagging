#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

RUN_STEP1="${ROOT}/run_teacher_hlt_only_weighted_5m1m1m.sh"
RUN_M2D005="${ROOT}/run_m2_joint_delta005_weighted_5m1m1m.sh"
RUN_M2D020="${ROOT}/run_m2_joint_delta020_weighted_5m1m1m.sh"
RUN_M4="${ROOT}/run_m4_recoteacher_s01_corrected_weighted_5m1m1m.sh"
RUN_M5="${ROOT}/run_m5_joint_s01_full_weighted_5m1m1m.sh"
RUN_M6="${ROOT}/run_m6_concat_stagea_corrected_weighted_5m1m1m.sh"
RUN_M9MID="${ROOT}/run_m9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m.sh"
RUN_M9HIGH="${ROOT}/run_m9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m.sh"
RUN_M12="${ROOT}/run_m12_dualreco_dualview_feat_noscale_weighted_5m1m1m.sh"
RUN_M15MID="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_5m1m1m.sh"
RUN_M15HIGH="${ROOT}/run_m15_dualreco_dualview_offdrop_high_weighted_5m1m1m.sh"
RUN_M16="${ROOT}/run_m16_dualreco_dualview_topk60_weighted_5m1m1m.sh"
RUN_M17="${ROOT}/run_m17_dualreco_dualview_antioverlap_weighted_5m1m1m.sh"
RUN_ANALYZE="${ROOT}/run_analyze_hlt_joint12_bin_gated_fusion_valsel_weighted_5m1m1m.sh"

QUEUE_ANALYZE="${QUEUE_ANALYZE:-0}"  # set 1 to queue analyzer as 14th job
INCLUDE_HLT_CANDIDATE="${INCLUDE_HLT_CANDIDATE:-1}"
STEP1_REF_NPZ="${STEP1_REF_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_weighted_5m1m1m/teacher_hlt_only_weighted_5m1m1m_seed0/results_step1_teacher_baseline.npz}"

jstep1=$(sbatch "${RUN_STEP1}" | awk '{print $4}')
j2a=$(sbatch "${RUN_M2D005}" | awk '{print $4}')
j2b=$(sbatch "${RUN_M2D020}" | awk '{print $4}')
j4=$(sbatch "${RUN_M4}" | awk '{print $4}')
j5=$(sbatch "${RUN_M5}" | awk '{print $4}')
j6=$(sbatch "${RUN_M6}" | awk '{print $4}')
j9m=$(sbatch "${RUN_M9MID}" | awk '{print $4}')
j9h=$(sbatch "${RUN_M9HIGH}" | awk '{print $4}')
j12=$(sbatch "${RUN_M12}" | awk '{print $4}')
j15m=$(sbatch "${RUN_M15MID}" | awk '{print $4}')
j15h=$(sbatch "${RUN_M15HIGH}" | awk '{print $4}')
j16=$(sbatch "${RUN_M16}" | awk '{print $4}')
j17=$(sbatch "${RUN_M17}" | awk '{print $4}')

dep="${jstep1}:${j2a}:${j2b}:${j4}:${j5}:${j6}:${j9m}:${j9h}:${j12}:${j15m}:${j15h}:${j16}:${j17}"

echo "Submitted 13 core jobs:"
echo "  step1_teacher_hlt=${jstep1}"
echo "  joint_delta005=${j2a}"
echo "  joint_delta020=${j2b}"
echo "  corrected_s01=${j4}"
echo "  joint_s01=${j5}"
echo "  concat_corrected=${j6}"
echo "  offdrop_mid=${j9m}"
echo "  offdrop_high=${j9h}"
echo "  dual_m12_noscale=${j12}"
echo "  dual_m15_offdrop_mid=${j15m}"
echo "  dual_m15_offdrop_high=${j15h}"
echo "  dual_m16_topk60=${j16}"
echo "  dual_m17_antioverlap=${j17}"
echo "Only analyzer will be queued with dependency chain: afterok:${dep}"

if [[ "${QUEUE_ANALYZE}" == "1" ]]; then
  export STEP1_REF_NPZ INCLUDE_HLT_CANDIDATE
  ja=$(sbatch --dependency="afterok:${dep}" "${RUN_ANALYZE}" | awk '{print $4}')
  echo "Queued analyzer as extra job: ${ja}"
fi
