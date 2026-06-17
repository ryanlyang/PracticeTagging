#!/usr/bin/env bash
#SBATCH --job-name=anM2bg
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze_m2j5_m2j20_bingated_only_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze_m2j5_m2j20_bingated_only_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE="${BASE:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"

M2J5_RUN_DIR="${M2J5_RUN_DIR:-${BASE}/model2_joint_delta005_weighted_5m1m1m/model2_joint_delta005_weighted_5m1m1m_seed0}"
M2J20_RUN_DIR="${M2J20_RUN_DIR:-${BASE}/model2_joint_delta020_weighted_5m1m1m/model2_joint_delta020_weighted_5m1m1m_seed0}"

M2J5_NPZ="${M2J5_NPZ:-${M2J5_RUN_DIR}/fusion_scores_val_test.npz}"
M2J20_NPZ="${M2J20_NPZ:-${M2J20_RUN_DIR}/fusion_scores_val_test.npz}"
STEP1_REF_NPZ="${STEP1_REF_NPZ:-${BASE}/teacher_hlt_only_weighted_5m1m1m/teacher_hlt_only_weighted_5m1m1m_seed0/results_step1_teacher_baseline.npz}"

TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta_joint}"
CANDIDATE_MODELS="${CANDIDATE_MODELS:-joint_delta_stage2,joint_delta_joint,hlt,joint_delta020_stage2,joint_delta020_joint}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
CALIBRATION="${CALIBRATION:-iso}"
HEAD_SELECT_MODE="${HEAD_SELECT_MODE:-first}"
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
EXPAND_PREPOST_VARIANTS="${EXPAND_PREPOST_VARIANTS:-0}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-${BASE}/analyze_m2j5_m2j20_weighted_5m1m1m/bin_gated}"
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

for p in "${M2J5_NPZ}" "${M2J20_NPZ}"; do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing required score file: ${p}" >&2
    exit 1
  fi
done

CMD=(
  python -u analyze_m2j5_m2j20_bingated_only_fusion.py
  --m2j5_npz "${M2J5_NPZ}"
  --m2j20_npz "${M2J20_NPZ}"
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
echo "M2J5/M2J20 Bin-Gated Only"
echo "TPRs:        ${TARGET_TPRS}"
echo "Out dir:     ${OUT_DIR}"
echo "Report JSON: ${REPORT_JSON}"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "Models:      ${CANDIDATE_MODELS}"
echo "M2J5 NPZ:    ${M2J5_NPZ}"
echo "M2J20 NPZ:   ${M2J20_NPZ}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
