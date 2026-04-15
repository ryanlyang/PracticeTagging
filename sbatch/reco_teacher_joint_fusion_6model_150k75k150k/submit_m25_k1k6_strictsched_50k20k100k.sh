#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/../.."

jid_k1=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m25_k1_strictsched_50k20k100k.sh | awk '{print $4}')
jid_k6=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m25_k6_strictsched_50k20k100k.sh | awk '{print $4}')

echo "Submitted:"
echo "  k1=${jid_k1}"
echo "  k6=${jid_k6}"
echo "Monitor with: squeue -u ${USER} | rg 'm25k(1|6)'"
