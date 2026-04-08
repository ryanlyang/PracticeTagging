#!/usr/bin/env bash
#SBATCH --job-name=an31bgv
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_bingated_valsel_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_bingated_valsel_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

FUSION_JSON_DEFAULT_CHECKPOINTS="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"
FUSION_JSON_DEFAULT_DOWNLOAD_A="download_checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"
FUSION_JSON_DEFAULT_DOWNLOAD_B="download_checkpoints/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"

if [[ -z "${FUSION_JSON:-}" ]]; then
  if [[ -f "${FUSION_JSON_DEFAULT_CHECKPOINTS}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_CHECKPOINTS}"
  elif [[ -f "${FUSION_JSON_DEFAULT_DOWNLOAD_A}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_DOWNLOAD_A}"
  elif [[ -f "${FUSION_JSON_DEFAULT_DOWNLOAD_B}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_DOWNLOAD_B}"
  else
    FUSION_JSON="${FUSION_JSON_DEFAULT_CHECKPOINTS}"
  fi
fi

TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.40}"
SEED="${SEED:-0}"
CALIBRATION="${CALIBRATION:-iso}"

# TPR-focused candidate pools from specialization atlas findings.
CANDIDATE_MODELS_TPR50="${CANDIDATE_MODELS_TPR50:-joint_delta,dual_m15_offdrop_mid,concat_corrected,hlt,dual_m17_antioverlap,dual_m16_topk60,offdrop_high}"
CANDIDATE_MODELS_TPR30="${CANDIDATE_MODELS_TPR30:-joint_delta,joint_s01,corrected_s01,offdrop_mid,dual_m12_noscale,dual_m16_topk60,dual_m15_offdrop_high,joint_delta020}"

SCORE_BAND_EDGES="${SCORE_BAND_EDGES:-0.0,0.8,0.9,1.0}"
DIST_NEAR_CUT="${DIST_NEAR_CUT:-0.0384}"
DIST_MID_LOW="${DIST_MID_LOW:-0.06285}"
DIST_MID_HIGH="${DIST_MID_HIGH:-0.07386}"

GLOBAL_MAX_ADD="${GLOBAL_MAX_ADD:-6}"
BIN_MAX_ADD="${BIN_MAX_ADD:-3}"
W_STEP="${W_STEP:-0.01}"
MIN_BIN_FIT="${MIN_BIN_FIT:-1200}"
MIN_GLOBAL_IMPROVE="${MIN_GLOBAL_IMPROVE:-1e-6}"
MIN_BIN_IMPROVE="${MIN_BIN_IMPROVE:-5e-6}"

OUT_DIR="${OUT_DIR:-}"
REPORT_JSON="${REPORT_JSON:-}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "${FUSION_JSON}" ]]; then
  echo "ERROR: FUSION_JSON not found: ${FUSION_JSON}" >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$(dirname "${FUSION_JSON}")/bin_gated_fusion_31_valsel"
fi

CMD=(
  python analyze_hlt_joint31_bin_gated_fusion.py
  --fusion_json "${FUSION_JSON}"
  --target_tprs "${TARGET_TPRS}"
  --anchor_model "${ANCHOR_MODEL}"
  --selection_mode "${SELECTION_MODE}"
  --candidate_models_tpr50 "${CANDIDATE_MODELS_TPR50}"
  --candidate_models_tpr30 "${CANDIDATE_MODELS_TPR30}"
  --router_cal_frac "${ROUTER_CAL_FRAC}"
  --seed "${SEED}"
  --calibration "${CALIBRATION}"
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
  --out_dir "${OUT_DIR}"
)

if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi

echo "============================================================"
echo "31-Model Bin-Gated Fusion (Val-Selected)"
echo "Fusion json: ${FUSION_JSON}"
echo "TPRs:        ${TARGET_TPRS}"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "Selection:   ${SELECTION_MODE}"
echo "Calibration: ${CALIBRATION}"
echo "Out dir:     ${OUT_DIR}"
echo "Cand@0.5:    ${CANDIDATE_MODELS_TPR50}"
echo "Cand@0.3:    ${CANDIDATE_MODELS_TPR30}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
