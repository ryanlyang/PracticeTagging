#!/usr/bin/env bash
set -euo pipefail

SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_m2_oracle_study_marginal_signal_150k75k150k.sh"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing script: ${SCRIPT}" >&2
  exit 1
fi

for v in A B C D E B_gen; do
  echo "Submitting variant ${v}"
  sbatch --export=ALL,ORACLE_VARIANT="${v}" "${SCRIPT}"
done

echo "Submitted oracle-study marginal diagnostics for variants: A B C D E B_gen"
