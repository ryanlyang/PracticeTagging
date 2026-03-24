#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M2="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m2_joint_delta005_150k75k150k.sh"
S_M3="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m3_recoteacher_s09_150k75k150k.sh"
S_M4="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_150k75k150k.sh"
S_M5="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m5_joint_s01_full_150k75k150k.sh"
S_M6="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m6_concat_stagea_corrected_150k75k150k.sh"
S_AN="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint6_fusion_150k75k150k.sh"

for s in "$S_M2" "$S_M3" "$S_M4" "$S_M5" "$S_M6" "$S_AN"; do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

jid_m2=$(sbatch "$S_M2" | awk '{print $4}')
jid_m3=$(sbatch "$S_M3" | awk '{print $4}')
jid_m4=$(sbatch "$S_M4" | awk '{print $4}')
jid_m5=$(sbatch "$S_M5" | awk '{print $4}')
jid_m6=$(sbatch "$S_M6" | awk '{print $4}')

jid_an=$(sbatch --dependency=afterok:${jid_m2}:${jid_m3}:${jid_m4}:${jid_m5}:${jid_m6} "$S_AN" | awk '{print $4}')

echo "Submitted model2 joint delta  : ${jid_m2}"
echo "Submitted model3 reco-teacher : ${jid_m3}"
echo "Submitted model4 corrected    : ${jid_m4}"
echo "Submitted model5 joint s01    : ${jid_m5}"
echo "Submitted model6 concat       : ${jid_m6}"
echo "Submitted six-model analysis  : ${jid_an} (afterok:${jid_m2}:${jid_m3}:${jid_m4}:${jid_m5}:${jid_m6})"
