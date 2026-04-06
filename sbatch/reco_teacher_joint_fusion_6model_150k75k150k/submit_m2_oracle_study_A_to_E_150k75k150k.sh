#!/usr/bin/env bash
set -euo pipefail

SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m2_joint_delta005_fulltrain_prog_unfreeze_oracle_study_150k75k150k.sh"
for v in A B C D E B_gen; do
  echo "Submitting ORACLE_VARIANT=${v}"
  ORACLE_VARIANT="${v}" sbatch "${SCRIPT}"
done

