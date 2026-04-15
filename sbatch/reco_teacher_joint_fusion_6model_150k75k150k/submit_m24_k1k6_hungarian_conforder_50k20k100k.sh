#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/../.."

jid_k1=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m24_k1_hungarian_conforder_50k20k100k.sh | awk '{print $4}')
jid_k6=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m24_k6_hungarian_conforder_50k20k100k.sh | awk '{print $4}')

echo "Submitted:"
echo "  k1=${jid_k1}"
echo "  k6=${jid_k6}"
echo "Monitor with: squeue -u ${USER} | rg 'm24k(1|6)'"
