#!/usr/bin/env bash
#SBATCH --job-name=an3stk
#SBATCH --partition=tier3
#SBATCH --time=6-00:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze3_stacked_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze3_stacked_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

# Inputs (override with sbatch --export=ALL,VAR=... if needed).
M4_RUN_DIR="${M4_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model4_recoteacher_s01_corrected_weighted_5m1m1m/model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0}"
M9MID_RUN_DIR="${M9MID_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0}"
M9HIGH_RUN_DIR="${M9HIGH_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0}"

M4_NPZ="${M4_NPZ:-${M4_RUN_DIR}/stageA_only_scores.npz}"
M9MID_NPZ="${M9MID_NPZ:-${M9MID_RUN_DIR}/stageA_residual_scores.npz}"
M9HIGH_NPZ="${M9HIGH_NPZ:-${M9HIGH_RUN_DIR}/stageA_residual_scores.npz}"

# Optional external HLT reference NPZ with preds_hlt_{val,test}.
HLT_NPZ="${HLT_NPZ:-}"

# Source choices (supports prejoint/postjoint aliases).
M4_SOURCES="${M4_SOURCES:-postjoint}"
M9MID_SOURCES="${M9MID_SOURCES:-prejoint,postjoint}"
M9HIGH_SOURCES="${M9HIGH_SOURCES:-prejoint,postjoint}"
INCLUDE_HLT="${INCLUDE_HLT:-1}"

TARGET_TPR="${TARGET_TPR:-0.50}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-fpr_at_tpr}"   # fpr_at_tpr|auc
BASE_CALIBRATION="${BASE_CALIBRATION:-iso}"  # raw|platt|iso

WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
WEIGHT_SEARCH_MODE="${WEIGHT_SEARCH_MODE:-auto}"  # auto|grid|dirichlet
MAX_WEIGHT_CANDIDATES="${MAX_WEIGHT_CANDIDATES:-250000}"
WEIGHT_RANDOM_SAMPLES="${WEIGHT_RANDOM_SAMPLES:-5000}"
WEIGHT_RANDOM_SEED="${WEIGHT_RANDOM_SEED:-52}"

STACK_FEATURES="${STACK_FEATURES:-logits_probs}"  # logits|probs|logits_probs
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0 30.0}"
STACK_CV="${STACK_CV:-5}"
STACK_MAX_ITER="${STACK_MAX_ITER:-5000}"
STACK_N_JOBS="${STACK_N_JOBS:-8}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/stacked_fusion_3_weighted_5m1m1m}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for p in "${M4_NPZ}" "${M9MID_NPZ}" "${M9HIGH_NPZ}"; do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing required score file: ${p}" >&2
    exit 1
  fi
done

CMD=(
  python analyze_hlt_joint3_stacked_logreg_fusion.py
  --m4_npz "${M4_NPZ}"
  --m9mid_npz "${M9MID_NPZ}"
  --m9high_npz "${M9HIGH_NPZ}"
  --m4_sources "${M4_SOURCES}"
  --m9mid_sources "${M9MID_SOURCES}"
  --m9high_sources "${M9HIGH_SOURCES}"
  --include_hlt "${INCLUDE_HLT}"
  --target_tpr "${TARGET_TPR}"
  --optimize_for "${OPTIMIZE_FOR}"
  --base_calibration "${BASE_CALIBRATION}"
  --weight_step "${WEIGHT_STEP}"
  --weight_search_mode "${WEIGHT_SEARCH_MODE}"
  --max_weight_candidates "${MAX_WEIGHT_CANDIDATES}"
  --weight_random_samples "${WEIGHT_RANDOM_SAMPLES}"
  --weight_random_seed "${WEIGHT_RANDOM_SEED}"
  --stack_features "${STACK_FEATURES}"
  --stack_Cs ${STACK_CS}
  --stack_cv "${STACK_CV}"
  --stack_max_iter "${STACK_MAX_ITER}"
  --stack_n_jobs "${STACK_N_JOBS}"
  --seed "${SEED}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${HLT_NPZ}" ]]; then
  CMD+=(--hlt_npz "${HLT_NPZ}")
fi

echo "============================================================"
echo "3-Run Stacked-LogReg Fusion (tier3 6d)"
echo "M4 NPZ:      ${M4_NPZ}"
echo "M9MID NPZ:   ${M9MID_NPZ}"
echo "M9HIGH NPZ:  ${M9HIGH_NPZ}"
echo "M4 sources:  ${M4_SOURCES}"
echo "M9mid srcs:  ${M9MID_SOURCES}"
echo "M9high srcs: ${M9HIGH_SOURCES}"
echo "Include HLT: ${INCLUDE_HLT}"
echo "Target TPR:  ${TARGET_TPR}"
echo "Optimize:    ${OPTIMIZE_FOR}"
echo "Calibration: ${BASE_CALIBRATION}"
echo "Out dir:     ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
