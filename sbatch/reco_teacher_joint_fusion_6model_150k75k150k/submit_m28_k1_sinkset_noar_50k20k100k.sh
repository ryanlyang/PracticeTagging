#!/usr/bin/env bash
set -euo pipefail
jid=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m28_k1_sinkset_noar_50k20k100k.sh | awk '{print $4}')
echo "Submitted m28 k1 sinkset-noar: ${jid}"
