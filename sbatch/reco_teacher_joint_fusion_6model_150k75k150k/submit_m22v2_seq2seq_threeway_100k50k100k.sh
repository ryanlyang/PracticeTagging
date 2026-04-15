#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/../.."

jid_phys=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m22v2_seq2seq_nexttoken_phys_100k50k100k.sh | awk '{print $4}')
jid_geom=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m22v2_seq2seq_nexttoken_geom_100k50k100k.sh | awk '{print $4}')
jid_full=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m22v2_seq2seq_nexttoken_full_100k50k100k.sh | awk '{print $4}')
jid_full_h=$(sbatch sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m22v2_seq2seq_nexttoken_full_hungarian_100k50k100k.sh | awk '{print $4}')

echo "Submitted:"
echo "  phys=${jid_phys}"
echo "  geom=${jid_geom}"
echo "  full=${jid_full}"
echo "  full_hungarian=${jid_full_h}"
echo "Monitor with: squeue -u ${USER} | rg 'm22v2(phys|geom|full|fullH)'"
