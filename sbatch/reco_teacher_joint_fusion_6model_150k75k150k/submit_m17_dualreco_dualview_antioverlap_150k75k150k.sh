#!/usr/bin/env bash
set -euo pipefail

ROOT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

jid=$(sbatch "${ROOT}/run_m17_dualreco_dualview_antioverlap_150k75k150k.sh" | awk '{print $4}')

echo "Submitted m17 dual-reco anti-overlap job: ${jid}"
