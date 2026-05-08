#!/usr/bin/env bash
#SBATCH --job-name=jt12fuse
#SBATCH --partition=tier3
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/build_joint12_fused_targets_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/build_joint12_fused_targets_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

SCORES_NPZ="${SCORES_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_150k150k300k_valsel/bin_gated_scores.npz}"
OUT_DIR="${OUT_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k}"
OVERALL_FAMILY="${OVERALL_FAMILY:-bin}"
REDUCTION="${REDUCTION:-mean}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CMD=(
  python build_toptag_fused_targets_from_joint12_bingated.py
  --scores_npz "${SCORES_NPZ}"
  --out_dir "${OUT_DIR}"
  --overall_family "${OVERALL_FAMILY}"
  --reduction "${REDUCTION}"
)

echo "============================================================"
echo "Build joint12 fused targets (top-tagging)"
echo "Scores: ${SCORES_NPZ}"
echo "Out:    ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${OUT_DIR}"
