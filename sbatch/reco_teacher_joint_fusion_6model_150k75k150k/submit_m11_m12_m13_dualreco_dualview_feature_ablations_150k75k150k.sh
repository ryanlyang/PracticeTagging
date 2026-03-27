#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

j11=$(sbatch "${ROOT}/run_m11_dualreco_dualview_feat_noangle_150k75k150k.sh" | awk '{print $4}')
j12=$(sbatch "${ROOT}/run_m12_dualreco_dualview_feat_noscale_150k75k150k.sh" | awk '{print $4}')
j13=$(sbatch "${ROOT}/run_m13_dualreco_dualview_feat_coreshape_150k75k150k.sh" | awk '{print $4}')

echo "Submitted dual-reco feature-ablation jobs:"
echo "  m11 no_angle : ${j11}"
echo "  m12 no_scale : ${j12}"
echo "  m13 core_shape: ${j13}"
