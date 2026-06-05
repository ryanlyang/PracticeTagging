#!/usr/bin/env bash
#SBATCH --job-name=jcSubCon
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_stacked_subset_contrib_1m250k1m_m2hybrid_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_stacked_subset_contrib_1m250k1m_m2hybrid_%j.err

set -euo pipefail

FUSION_DIR="${FUSION_DIR:-checkpoints/jetclass_joint_dualview/fusion_reports/twelve_model_1m250k1m_m2hybrid_stacked_acc}"
SCORES_NPZ="${SCORES_NPZ:-${FUSION_DIR}/fusion_scores.npz}"
REPORT_JSON="${REPORT_JSON:-${FUSION_DIR}/report.json}"
OUT_DIR="${OUT_DIR:-${FUSION_DIR}/subset_contribution_analysis}"
SCRIPT="${SCRIPT:-$(pwd)/analyze_jetclass_stacked_subset_contributions.py}"

FEATURE_MODE="${FEATURE_MODE:-logits_probs}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-4}"
TOP_K="${TOP_K:-30}"
GREEDY_STEPS="${GREEDY_STEPS:-8}"
CV="${CV:-5}"
MAX_ITER="${MAX_ITER:-2000}"
N_JOBS="${N_JOBS:-1}"
SKIP_AUC="${SKIP_AUC:-1}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

CMD=(
  python -u "${SCRIPT}"
  --scores_npz "${SCORES_NPZ}"
  --report_json "${REPORT_JSON}"
  --out_dir "${OUT_DIR}"
  --feature_mode "${FEATURE_MODE}"
  --max_combo_size "${MAX_COMBO_SIZE}"
  --top_k "${TOP_K}"
  --greedy_steps "${GREEDY_STEPS}"
  --cv "${CV}"
  --max_iter "${MAX_ITER}"
  --n_jobs "${N_JOBS}"
)

if [[ "${SKIP_AUC}" == "1" || "${SKIP_AUC}" == "true" || "${SKIP_AUC}" == "TRUE" ]]; then
  CMD+=(--skip_auc)
fi

echo "============================================================"
echo "JetClass stacked subset contribution analysis"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Fusion dir: ${FUSION_DIR}"
echo "Scores:     ${SCORES_NPZ}"
echo "Report:     ${REPORT_JSON}"
echo "Out dir:    ${OUT_DIR}"
echo "Features:   ${FEATURE_MODE}"
echo "Combos:     up to ${MAX_COMBO_SIZE}"
echo "CV:         ${CV}"
echo "Skip AUC:   ${SKIP_AUC}"
echo "============================================================"
echo " ${CMD[*]}"
"${CMD[@]}"

