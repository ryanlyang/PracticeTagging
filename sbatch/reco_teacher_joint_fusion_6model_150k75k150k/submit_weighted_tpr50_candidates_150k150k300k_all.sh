#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

sbatch "${ROOT}/run_m15_dualreco_dualview_offdrop_mid_weighted_150k150k300k.sh"
sbatch "${ROOT}/run_m6_concat_stagea_corrected_weighted_150k150k300k.sh"
sbatch "${ROOT}/run_m17_dualreco_dualview_antioverlap_weighted_150k150k300k.sh"
sbatch "${ROOT}/run_m16_dualreco_dualview_topk60_weighted_150k150k300k.sh"
sbatch "${ROOT}/run_m9_stageA_residual_hlt_offdrop_high_weighted_150k150k300k.sh"
