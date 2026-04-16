#!/usr/bin/env bash
set -euo pipefail
jid=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m27_k1_greedybip_50k20k100k.sh | awk '{print $4}')
echo "Submitted m27 k1 greedybip: ${jid}"
