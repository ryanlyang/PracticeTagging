#!/usr/bin/env bash
#SBATCH --job-name=an9bg
#SBATCH --partition=tier3
#SBATCH --time=6-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze9_bingated_only_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze9_bingated_only_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE="${BASE:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"

M4_RUN_DIR="${M4_RUN_DIR:-${BASE}/model4_recoteacher_s01_corrected_weighted_5m1m1m/model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0}"
M5_RUN_DIR="${M5_RUN_DIR:-${BASE}/model5_joint_s01_full_weighted_5m1m1m/model5_joint_s01_full_weighted_5m1m1m_seed0}"
M9MID_RUN_DIR="${M9MID_RUN_DIR:-${BASE}/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0}"
M9HIGH_RUN_DIR="${M9HIGH_RUN_DIR:-${BASE}/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0}"
M12_RUN_DIR="${M12_RUN_DIR:-${BASE}/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_from_recoonly/model12_dualreco_dualview_feat_noscale_weighted_5m1m1m_seed0_from_recoonly}"
M15MID_RUN_DIR="${M15MID_RUN_DIR:-${BASE}/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_from_recoonly/model15_dualreco_dualview_offdrop_mid_weighted_5m1m1m_seed0_from_recoonly}"
M15HIGH_RUN_DIR="${M15HIGH_RUN_DIR:-${BASE}/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_from_recoonly/model15_dualreco_dualview_offdrop_high_weighted_5m1m1m_seed0_from_recoonly}"
M16_RUN_DIR="${M16_RUN_DIR:-${BASE}/model16_dualreco_dualview_topk60_weighted_5m1m1m_from_recoonly/model16_dualreco_dualview_topk60_weighted_5m1m1m_seed0_from_recoonly}"
M17_RUN_DIR="${M17_RUN_DIR:-${BASE}/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_recoonly/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_recoonly}"

M4_NPZ="${M4_NPZ:-${M4_RUN_DIR}/stageA_only_scores.npz}"
M5_NPZ="${M5_NPZ:-${M5_RUN_DIR}/fusion_scores_val_test.npz}"
M9MID_NPZ="${M9MID_NPZ:-${M9MID_RUN_DIR}/stageA_residual_scores.npz}"
M9HIGH_NPZ="${M9HIGH_NPZ:-${M9HIGH_RUN_DIR}/stageA_residual_scores.npz}"
M12_NPZ="${M12_NPZ:-${M12_RUN_DIR}/dualreco_dualview_scores.npz}"
M15MID_NPZ="${M15MID_NPZ:-${M15MID_RUN_DIR}/dualreco_dualview_scores.npz}"
M15HIGH_NPZ="${M15HIGH_NPZ:-${M15HIGH_RUN_DIR}/dualreco_dualview_scores.npz}"
M16_NPZ="${M16_NPZ:-${M16_RUN_DIR}/dualreco_dualview_scores.npz}"
M17_NPZ="${M17_NPZ:-${M17_RUN_DIR}/dualreco_dualview_scores.npz}"

STEP1_REF_NPZ="${STEP1_REF_NPZ:-${BASE}/teacher_hlt_only_weighted_5m1m1m/teacher_hlt_only_weighted_5m1m1m_seed0/results_step1_teacher_baseline.npz}"

TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
ANCHOR_MODEL="${ANCHOR_MODEL:-offdrop_mid}"
CANDIDATE_MODELS="${CANDIDATE_MODELS:-corrected_s01,joint_s01,offdrop_mid,offdrop_high,dual_m12_noscale,dual_m15_offdrop_mid,dual_m15_offdrop_high,dual_m16_topk60,dual_m17_antioverlap,hlt}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
CALIBRATION="${CALIBRATION:-iso}"
HEAD_SELECT_MODE="${HEAD_SELECT_MODE:-best_val_fpr}"
HEAD_SELECT_TPR="${HEAD_SELECT_TPR:-0.50}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.40}"
SCORE_BAND_EDGES="${SCORE_BAND_EDGES:-0.0,0.8,0.9,1.0}"
DIST_NEAR_CUT="${DIST_NEAR_CUT:-0.0384}"
DIST_MID_LOW="${DIST_MID_LOW:-0.06285}"
DIST_MID_HIGH="${DIST_MID_HIGH:-0.07386}"
GLOBAL_MAX_ADD="${GLOBAL_MAX_ADD:-8}"
BIN_MAX_ADD="${BIN_MAX_ADD:-6}"
W_STEP="${W_STEP:-0.0025}"
MIN_BIN_FIT="${MIN_BIN_FIT:-2000}"
MIN_GLOBAL_IMPROVE="${MIN_GLOBAL_IMPROVE:-2e-7}"
MIN_BIN_IMPROVE="${MIN_BIN_IMPROVE:-1e-6}"
EXPAND_PREPOST_VARIANTS="${EXPAND_PREPOST_VARIANTS:-1}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-${BASE}/analyze9_finished_weighted_5m1m1m/bin_gated}"
REPORT_JSON="${REPORT_JSON:-${OUT_DIR}/bin_gated_report.json}"

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
  "${M5_NPZ}" \
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
  python -u analyze_hlt_joint9_bingated_only_fusion.py
  --m4_npz "${M4_NPZ}"
  --m5_npz "${M5_NPZ}"
  --m9mid_npz "${M9MID_NPZ}"
  --m9high_npz "${M9HIGH_NPZ}"
  --m12_npz "${M12_NPZ}"
  --m15mid_npz "${M15MID_NPZ}"
  --m15high_npz "${M15HIGH_NPZ}"
  --m16_npz "${M16_NPZ}"
  --m17_npz "${M17_NPZ}"
  --step1_ref_npz "${STEP1_REF_NPZ}"
  --target_tprs "${TARGET_TPRS}"
  --anchor_model "${ANCHOR_MODEL}"
  --candidate_models "${CANDIDATE_MODELS}"
  --selection_mode "${SELECTION_MODE}"
  --calibration "${CALIBRATION}"
  --head_select_mode "${HEAD_SELECT_MODE}"
  --head_select_tpr "${HEAD_SELECT_TPR}"
  --router_cal_frac "${ROUTER_CAL_FRAC}"
  --score_band_edges "${SCORE_BAND_EDGES}"
  --dist_near_cut "${DIST_NEAR_CUT}"
  --dist_mid_low "${DIST_MID_LOW}"
  --dist_mid_high "${DIST_MID_HIGH}"
  --global_max_add "${GLOBAL_MAX_ADD}"
  --bin_max_add "${BIN_MAX_ADD}"
  --w_step "${W_STEP}"
  --min_bin_fit "${MIN_BIN_FIT}"
  --min_global_improve "${MIN_GLOBAL_IMPROVE}"
  --min_bin_improve "${MIN_BIN_IMPROVE}"
  --expand_prepost_variants "${EXPAND_PREPOST_VARIANTS}"
  --seed "${SEED}"
  --out_dir "${OUT_DIR}"
  --report_json "${REPORT_JSON}"
)

echo "============================================================"
echo "Analyze9 Bin-Gated Only (m5 added to Analyze8 finished set)"
echo "TPRs:        ${TARGET_TPRS}"
echo "Out dir:     ${OUT_DIR}"
echo "Report JSON: ${REPORT_JSON}"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "Models:      ${CANDIDATE_MODELS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
