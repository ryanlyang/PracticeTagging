#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/../.."

jid_tag=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m23_multihyp_k6_tag_50k20k100k.sh | awk '{print $4}')
jid_reco=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m23_multihyp_k6_reco_50k20k100k.sh | awk '{print $4}')
jid_hyb=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m23_multihyp_k6_hybrid_50k20k100k.sh | awk '{print $4}')

echo "Submitted:"
echo "  tag=${jid_tag}"
echo "  reco=${jid_reco}"
echo "  hybrid=${jid_hyb}"
echo "Monitor with: squeue -u ${USER} | rg 'm23k6(tag|reco|hyb)'"
