#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

jid_low=$(sbatch "${ROOT}/run_m15_dualreco_dualview_offdrop_low_150k75k150k.sh" | awk '{print $4}')
jid_mid=$(sbatch "${ROOT}/run_m15_dualreco_dualview_offdrop_mid_150k75k150k.sh" | awk '{print $4}')
jid_high=$(sbatch "${ROOT}/run_m15_dualreco_dualview_offdrop_high_150k75k150k.sh" | awk '{print $4}')

echo "Submitted m15 dual-reco dualview jobs:"
echo "  low : ${jid_low}"
echo "  mid : ${jid_mid}"
echo "  high: ${jid_high}"
