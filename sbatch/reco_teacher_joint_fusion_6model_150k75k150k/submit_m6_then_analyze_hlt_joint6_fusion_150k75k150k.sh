#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M6="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m6_concat_stagea_corrected_150k75k150k.sh"
S_AN="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint6_fusion_150k75k150k.sh"

for s in "$S_M6" "$S_AN"; do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

jid_m6=$(sbatch "$S_M6" | awk '{print $4}')
jid_an=$(sbatch --dependency=afterok:${jid_m6} "$S_AN" | awk '{print $4}')

echo "Submitted model6 concat run : ${jid_m6}"
echo "Submitted six-model analysis: ${jid_an} (afterok:${jid_m6})"
