#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

jid40=$(sbatch "${ROOT}/run_m16_dualreco_dualview_topk40_150k75k150k.sh" | awk '{print $4}')
jid60=$(sbatch "${ROOT}/run_m16_dualreco_dualview_topk60_150k75k150k.sh" | awk '{print $4}')
jid80=$(sbatch "${ROOT}/run_m16_dualreco_dualview_topk80_150k75k150k.sh" | awk '{print $4}')

echo "Submitted m16 dual-reco top-k jobs:"
echo "  k40: ${jid40}"
echo "  k60: ${jid60}"
echo "  k80: ${jid80}"
