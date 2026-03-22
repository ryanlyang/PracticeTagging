#!/usr/bin/env bash
#SBATCH -J fusNSdbg
#SBATCH -p debug
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o offline_reconstructor_logs/fusion_analysis_normal_special/fusion_ns_%j.out
#SBATCH -e offline_reconstructor_logs/fusion_analysis_normal_special/fusion_ns_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/fusion_analysis_normal_special}"
mkdir -p "${LOG_DIR}"

# Defaults match your recent runs; override at submit time if needed.
NORMAL_RUN_DIR="${NORMAL_RUN_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_normal_extended_combo/stagec_normal_extval100k_exttest300k_combo_seed0}"
SPECIAL_RUN_DIR="${SPECIAL_RUN_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_specialist_against_normal/stagec_specialist_pushhard_anchorNormal_comboSel_union_seed0}"

TARGET_TPR="${TARGET_TPR:-0.50}"
WEIGHT_STEP="${WEIGHT_STEP:-0.01}"
HARD_TPR_MIN="${HARD_TPR_MIN:-0.30}"
HARD_TPR_MAX="${HARD_TPR_MAX:-0.60}"
HARD_TPR_STEP="${HARD_TPR_STEP:-0.01}"
HARD_TARGET_TOL="${HARD_TARGET_TOL:-0.003}"
TOP_K="${TOP_K:-100}"

SAVE_DIR="${SAVE_DIR:-download_checkpoints/fusion_analysis_normal_special_target50}"

# If local defaults are missing, fallback to copied download_checkpoints paths.
if [[ ! -f "${NORMAL_RUN_DIR}/results.npz" ]]; then
  ALT_NORMAL="download_checkpoints/stagec_normal_extval100k_exttest300k_combo_seed0"
  if [[ -f "${ALT_NORMAL}/results.npz" ]]; then
    NORMAL_RUN_DIR="${ALT_NORMAL}"
  fi
fi
if [[ ! -f "${SPECIAL_RUN_DIR}/results.npz" ]]; then
  ALT_SPECIAL="download_checkpoints/stagec_specialist_pushhard_anchorNormal_comboSel_union_seed0"
  if [[ -f "${ALT_SPECIAL}/results.npz" ]]; then
    SPECIAL_RUN_DIR="${ALT_SPECIAL}"
  fi
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Normal+Special fusion analysis"
echo "Normal run dir : ${NORMAL_RUN_DIR}"
echo "Special run dir: ${SPECIAL_RUN_DIR}"
echo "Target TPR     : ${TARGET_TPR}"
echo "Save dir       : ${SAVE_DIR}"
echo "============================================================"

python analyze_normal_special_fusion_strategies.py \
  --normal_run_dir "${NORMAL_RUN_DIR}" \
  --special_run_dir "${SPECIAL_RUN_DIR}" \
  --target_tpr "${TARGET_TPR}" \
  --weight_step "${WEIGHT_STEP}" \
  --hard_tpr_min "${HARD_TPR_MIN}" \
  --hard_tpr_max "${HARD_TPR_MAX}" \
  --hard_tpr_step "${HARD_TPR_STEP}" \
  --hard_target_tol "${HARD_TARGET_TOL}" \
  --top_k "${TOP_K}" \
  --save_dir "${SAVE_DIR}"

echo
echo "Done."
echo "Summary: ${SAVE_DIR}/fusion_analysis_summary.json"
echo "Best table: ${SAVE_DIR}/fusion_best_summary.tsv"
