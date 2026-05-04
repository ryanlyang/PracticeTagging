#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

RUN_M15="${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_150k150k300k.sh"
RUN_M6="${ROOT}/run_m6_concat_stagea_corrected_weighted_150k150k300k.sh"
RUN_M17="${ROOT}/run_m17_dualreco_dualview_antioverlap_weighted_150k150k300k.sh"
RUN_M16="${ROOT}/run_m16_dualreco_dualview_topk60_weighted_150k150k300k.sh"
RUN_M9="${ROOT}/run_m9_stageA_residual_hlt_offdrop_high_weighted_150k150k300k.sh"
RUN_ANALYZE="${ROOT}/run_analyze_hlt_joint5_bin_gated_fusion_valsel_weighted_150k150k300k.sh"

# Optional:
#   DEP_JOB_IDS="21200001:21200002:21200003:21200004:21200005" bash .../submit_...sh
# If DEP_JOB_IDS is set, no training jobs are submitted here; analysis is queued with that dependency.
if [[ -n "${DEP_JOB_IDS:-}" ]]; then
  dep="${DEP_JOB_IDS}"
  echo "Using provided dependency chain: ${dep}"
else
  j15=$(sbatch "${RUN_M15}" | awk '{print $4}')
  j6=$(sbatch "${RUN_M6}" | awk '{print $4}')
  j17=$(sbatch "${RUN_M17}" | awk '{print $4}')
  j16=$(sbatch "${RUN_M16}" | awk '{print $4}')
  j9=$(sbatch "${RUN_M9}" | awk '{print $4}')
  dep="${j15}:${j6}:${j17}:${j16}:${j9}"
  echo "Submitted training jobs:"
  echo "  m15=${j15}"
  echo "  m6=${j6}"
  echo "  m17=${j17}"
  echo "  m16=${j16}"
  echo "  m9=${j9}"
fi

ja=$(sbatch --dependency="afterok:${dep}" "${RUN_ANALYZE}" | awk '{print $4}')
echo "Queued dependent analysis job: ${ja}"
echo "Dependency: afterok:${dep}"
