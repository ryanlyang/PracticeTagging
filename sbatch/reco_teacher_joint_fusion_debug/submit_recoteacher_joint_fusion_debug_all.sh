#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S1="sbatch/reco_teacher_joint_fusion_debug/run_recoteacher_stageAonly_s09delta_75k25k150k_debug.sh"
S2="sbatch/reco_teacher_joint_fusion_debug/run_joint_unmergeonly_75k25k150k_debug.sh"
S3="sbatch/reco_teacher_joint_fusion_debug/run_analyze_hlt_joint_recoteacher_fusion_debug.sh"

for s in "$S1" "$S2" "$S3"; do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

jid_stagea=$(sbatch "$S1" | awk '{print $4}')
jid_joint=$(sbatch "$S2" | awk '{print $4}')
jid_analysis=$(sbatch --dependency=afterok:${jid_stagea}:${jid_joint} "$S3" | awk '{print $4}')

echo "Submitted StageA-only RecoTeacher: ${jid_stagea}"
echo "Submitted separate Joint run     : ${jid_joint}"
echo "Submitted fusion analysis        : ${jid_analysis} (afterok:${jid_stagea}:${jid_joint})"
