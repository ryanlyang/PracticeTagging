#!/usr/bin/env bash
#SBATCH --job-name=an8stbg
#SBATCH --partition=tier3
#SBATCH --time=6-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze8_stacked_bingated_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze8_stacked_bingated_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE="${BASE:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"

M4_RUN_DIR="${M4_RUN_DIR:-${BASE}/model4_recoteacher_s01_corrected_weighted_5m1m1m/model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0}"
M9MID_RUN_DIR="${M9MID_RUN_DIR:-${BASE}/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0}"
M9HIGH_RUN_DIR="${M9HIGH_RUN_DIR:-${BASE}/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0}"
M12_RUN_DIR="${M12_RUN_DIR:-${BASE}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_from_recoonly/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_from_recoonly}"
M15MID_RUN_DIR="${M15MID_RUN_DIR:-${BASE}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_from_recoonly/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_from_recoonly}"
M15HIGH_RUN_DIR="${M15HIGH_RUN_DIR:-${BASE}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_from_recoonly/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_from_recoonly}"
M16_RUN_DIR="${M16_RUN_DIR:-${BASE}/model16_dualreco_dualview_topk60_weighted_5m1m1m_from_recoonly/model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_from_recoonly}"
M17_RUN_DIR="${M17_RUN_DIR:-${BASE}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_recoonly/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_recoonly}"

M4_NPZ="${M4_NPZ:-${M4_RUN_DIR}/stageA_only_scores.npz}"
M9MID_NPZ="${M9MID_NPZ:-${M9MID_RUN_DIR}/stageA_residual_scores.npz}"
M9HIGH_NPZ="${M9HIGH_NPZ:-${M9HIGH_RUN_DIR}/stageA_residual_scores.npz}"
M12_NPZ="${M12_NPZ:-${M12_RUN_DIR}/dualreco_dualview_scores.npz}"
M15MID_NPZ="${M15MID_NPZ:-${M15MID_RUN_DIR}/dualreco_dualview_scores.npz}"
M15HIGH_NPZ="${M15HIGH_NPZ:-${M15HIGH_RUN_DIR}/dualreco_dualview_scores.npz}"
M16_NPZ="${M16_NPZ:-${M16_RUN_DIR}/dualreco_dualview_scores.npz}"
M17_NPZ="${M17_NPZ:-${M17_RUN_DIR}/dualreco_dualview_scores.npz}"

HLT_NPZ="${HLT_NPZ:-}"
STEP1_REF_NPZ="${STEP1_REF_NPZ:-${BASE}/teacher_hlt_only_weighted_5m1m1m/teacher_hlt_only_weighted_5m1m1m_seed0/results_step1_teacher_baseline.npz}"

TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-fpr_at_tpr}"
BASE_CALIBRATION="${BASE_CALIBRATION:-iso}"
HEAD_SELECT_MODE="${HEAD_SELECT_MODE:-best_val_fpr}"
HEAD_SELECT_TPR="${HEAD_SELECT_TPR:-0.50}"

WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
WEIGHT_SEARCH_MODE="${WEIGHT_SEARCH_MODE:-auto}"
MAX_WEIGHT_CANDIDATES="${MAX_WEIGHT_CANDIDATES:-250000}"
WEIGHT_RANDOM_SAMPLES="${WEIGHT_RANDOM_SAMPLES:-20000}"
WEIGHT_RANDOM_SEED="${WEIGHT_RANDOM_SEED:-52}"

STACK_FEATURES="${STACK_FEATURES:-logits_probs}"
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0 30.0}"
STACK_CV="${STACK_CV:-5}"
STACK_MAX_ITER="${STACK_MAX_ITER:-5000}"
STACK_N_JOBS="${STACK_N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
SEED="${SEED:-0}"

BIN_ANCHOR_MODEL="${BIN_ANCHOR_MODEL:-offdrop_mid}"
BIN_CANDIDATE_MODELS="${BIN_CANDIDATE_MODELS:-corrected_s01,offdrop_mid,offdrop_high,dual_m12_noscale,dual_m15_offdrop_mid,dual_m15_offdrop_high,dual_m16_topk60,dual_m17_antioverlap,hlt}"
BIN_EXPAND_PREPOST_VARIANTS="${BIN_EXPAND_PREPOST_VARIANTS:-1}"
BIN_SELECTION_MODE="${BIN_SELECTION_MODE:-valsel}"
BIN_CALIBRATION="${BIN_CALIBRATION:-iso}"
BIN_HEAD_SELECT_MODE="${BIN_HEAD_SELECT_MODE:-best_val_fpr}"
BIN_HEAD_SELECT_TPR="${BIN_HEAD_SELECT_TPR:-0.50}"
BIN_ROUTER_CAL_FRAC="${BIN_ROUTER_CAL_FRAC:-0.40}"
BIN_SCORE_BAND_EDGES="${BIN_SCORE_BAND_EDGES:-0.0,0.8,0.9,1.0}"
BIN_DIST_NEAR_CUT="${BIN_DIST_NEAR_CUT:-0.0384}"
BIN_DIST_MID_LOW="${BIN_DIST_MID_LOW:-0.06285}"
BIN_DIST_MID_HIGH="${BIN_DIST_MID_HIGH:-0.07386}"
BIN_GLOBAL_MAX_ADD="${BIN_GLOBAL_MAX_ADD:-8}"
BIN_BIN_MAX_ADD="${BIN_BIN_MAX_ADD:-6}"
BIN_W_STEP="${BIN_W_STEP:-0.0025}"
BIN_MIN_BIN_FIT="${BIN_MIN_BIN_FIT:-2000}"
BIN_MIN_GLOBAL_IMPROVE="${BIN_MIN_GLOBAL_IMPROVE:-2e-7}"
BIN_MIN_BIN_IMPROVE="${BIN_MIN_BIN_IMPROVE:-1e-6}"

SKIP_STACKED="${SKIP_STACKED:-0}"
SKIP_BIN_GATED="${SKIP_BIN_GATED:-0}"
OUT_DIR="${OUT_DIR:-${BASE}/analyze8_finished_weighted_5m1m1m}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for p in \
  "${M4_NPZ}" \
  "${M9MID_NPZ}" \
  "${M9HIGH_NPZ}" \
  "${M12_NPZ}" \
  "${M15MID_NPZ}" \
  "${M15HIGH_NPZ}" \
  "${M16_NPZ}" \
  "${M17_NPZ}"
do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing required score file: ${p}" >&2
    exit 1
  fi
done

CMD=(
  python -u analyze_hlt_joint8_stacked_logreg_bingated_fusion.py
  --m4_npz "${M4_NPZ}"
  --m9mid_npz "${M9MID_NPZ}"
  --m9high_npz "${M9HIGH_NPZ}"
  --m12_npz "${M12_NPZ}"
  --m15mid_npz "${M15MID_NPZ}"
  --m15high_npz "${M15HIGH_NPZ}"
  --m16_npz "${M16_NPZ}"
  --m17_npz "${M17_NPZ}"
  --step1_ref_npz "${STEP1_REF_NPZ}"
  --target_tprs "${TARGET_TPRS}"
  --optimize_for "${OPTIMIZE_FOR}"
  --base_calibration "${BASE_CALIBRATION}"
  --head_select_mode "${HEAD_SELECT_MODE}"
  --head_select_tpr "${HEAD_SELECT_TPR}"
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
  --bin_anchor_model "${BIN_ANCHOR_MODEL}"
  --bin_candidate_models "${BIN_CANDIDATE_MODELS}"
  --bin_expand_prepost_variants "${BIN_EXPAND_PREPOST_VARIANTS}"
  --bin_selection_mode "${BIN_SELECTION_MODE}"
  --bin_calibration "${BIN_CALIBRATION}"
  --bin_head_select_mode "${BIN_HEAD_SELECT_MODE}"
  --bin_head_select_tpr "${BIN_HEAD_SELECT_TPR}"
  --bin_router_cal_frac "${BIN_ROUTER_CAL_FRAC}"
  --bin_score_band_edges "${BIN_SCORE_BAND_EDGES}"
  --bin_dist_near_cut "${BIN_DIST_NEAR_CUT}"
  --bin_dist_mid_low "${BIN_DIST_MID_LOW}"
  --bin_dist_mid_high "${BIN_DIST_MID_HIGH}"
  --bin_global_max_add "${BIN_GLOBAL_MAX_ADD}"
  --bin_bin_max_add "${BIN_BIN_MAX_ADD}"
  --bin_w_step "${BIN_W_STEP}"
  --bin_min_bin_fit "${BIN_MIN_BIN_FIT}"
  --bin_min_global_improve "${BIN_MIN_GLOBAL_IMPROVE}"
  --bin_min_bin_improve "${BIN_MIN_BIN_IMPROVE}"
  --skip_stacked "${SKIP_STACKED}"
  --skip_bin_gated "${SKIP_BIN_GATED}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${HLT_NPZ}" ]]; then
  CMD+=(--hlt_npz "${HLT_NPZ}")
fi

echo "============================================================"
echo "Analyze8 Finished Models: Stacked LogReg + Bin-Gated"
echo "TPRs:        ${TARGET_TPRS}"
echo "Out dir:     ${OUT_DIR}"
echo "Stack cal:   ${BASE_CALIBRATION}"
echo "Head select: ${HEAD_SELECT_MODE} @TPR=${HEAD_SELECT_TPR}"
echo "Bin anchor:  ${BIN_ANCHOR_MODEL}"
echo "Bin models:  ${BIN_CANDIDATE_MODELS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
