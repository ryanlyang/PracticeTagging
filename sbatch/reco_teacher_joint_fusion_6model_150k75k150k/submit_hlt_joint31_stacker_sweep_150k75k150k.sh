#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_SCRIPT="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint31_stacker_variant_150k75k150k.sh"

if [[ $# -gt 0 ]]; then
  VARIANTS=("$@")
else
  VARIANTS=(
    baseline_meta
    sparse_only
    tiny_mlp_only
    moe_only
    sparse_mlp
    sparse_moe
    mlp_moe
    all_stackers
  )
fi

echo "Submitting 31-model fusion stacker sweep..."
for v in "${VARIANTS[@]}"; do
  short="${v}"
  short="${short/baseline_meta/base}"
  short="${short/tiny_mlp_only/mlp}"
  short="${short/all_stackers/all}"
  short="${short/sparse_only/sp}"
  short="${short/moe_only/moe}"
  short="${short/sparse_mlp/spmlp}"
  short="${short/sparse_moe/spmoe}"
  short="${short/mlp_moe/mlpmoe}"

  jid=$(sbatch --job-name="an31_${short}" --export=ALL,FUSION_VARIANT="${v}" "${RUN_SCRIPT}" | awk '{print $4}')
  echo "  ${v} -> ${jid}"
done

echo "Submitted ${#VARIANTS[@]} jobs."
