#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S11="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m11_recoteacher_s01_corrected_feat_noangle_150k75k150k.sh"
S12="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m12_recoteacher_s01_corrected_feat_noscale_150k75k150k.sh"
S13="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m13_recoteacher_s01_corrected_feat_coreshape_150k75k150k.sh"

for s in "$S11" "$S12" "$S13"; do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

j11=$(sbatch "$S11" | awk '{print $4}')
j12=$(sbatch "$S12" | awk '{print $4}')
j13=$(sbatch "$S13" | awk '{print $4}')

echo "Submitted feature-ablation m4-style jobs:"
echo "  m11 (no_angle): ${j11}"
echo "  m12 (no_scale): ${j12}"
echo "  m13 (core_shape): ${j13}"
