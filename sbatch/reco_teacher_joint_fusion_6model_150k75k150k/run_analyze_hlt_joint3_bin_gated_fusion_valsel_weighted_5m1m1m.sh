#!/usr/bin/env bash
#SBATCH --job-name=an3bgw
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=320G
#SBATCH --time=10-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze3_bingated_valsel_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze3_bingated_valsel_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

M4_RUN_DIR="${M4_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model4_recoteacher_s01_corrected_weighted_5m1m1m/model4_recoteacher_s01_corrected_weighted_5m1m1m_seed0}"
M9MID_RUN_DIR="${M9MID_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_mid_weighted_5m1m1m_seed0}"
M9HIGH_RUN_DIR="${M9HIGH_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m/model9_stageA_residual_hlt_offdrop_high_weighted_5m1m1m_seed0}"

M4_NPZ="${M4_NPZ:-${M4_RUN_DIR}/stageA_only_scores.npz}"
M9MID_NPZ="${M9MID_NPZ:-${M9MID_RUN_DIR}/stageA_residual_scores.npz}"
M9HIGH_NPZ="${M9HIGH_NPZ:-${M9HIGH_RUN_DIR}/stageA_residual_scores.npz}"

ANCHOR_MODEL="${ANCHOR_MODEL:-offdrop_mid}"
TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
CALIBRATION="${CALIBRATION:-iso}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.40}"
SEED="${SEED:-0}"
HEAD_SELECT_MODE="${HEAD_SELECT_MODE:-best_val_fpr}"
HEAD_SELECT_TPR="${HEAD_SELECT_TPR:-0.50}"

CANDIDATE_MODELS_ALL="${CANDIDATE_MODELS_ALL:-corrected_s01,offdrop_mid,offdrop_high}"
INCLUDE_HLT_CANDIDATE="${INCLUDE_HLT_CANDIDATE:-1}"
STEP1_REF_NPZ="${STEP1_REF_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_weighted_5m1m1m/teacher_hlt_only_weighted_5m1m1m_seed0/results_step1_teacher_baseline.npz}"

# Fine-grained search.
GLOBAL_MAX_ADD="${GLOBAL_MAX_ADD:-8}"
BIN_MAX_ADD="${BIN_MAX_ADD:-6}"
W_STEP="${W_STEP:-0.0025}"
MIN_BIN_FIT="${MIN_BIN_FIT:-2000}"
MIN_GLOBAL_IMPROVE="${MIN_GLOBAL_IMPROVE:-2e-7}"
MIN_BIN_IMPROVE="${MIN_BIN_IMPROVE:-1e-6}"

SCORE_BAND_EDGES="${SCORE_BAND_EDGES:-0.0,0.8,0.9,1.0}"
DIST_NEAR_CUT="${DIST_NEAR_CUT:-0.0384}"
DIST_MID_LOW="${DIST_MID_LOW:-0.06285}"
DIST_MID_HIGH="${DIST_MID_HIGH:-0.07386}"

OUT_DIR="${OUT_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_3_weighted_5m1m1m_valsel}"
FUSION_JSON="${FUSION_JSON:-${OUT_DIR}/fusion_hlt_joint3_weighted_5m1m1m.json}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUT_DIR}"

for p in "${M4_NPZ}" "${M9MID_NPZ}" "${M9HIGH_NPZ}"
do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing required score file: ${p}" >&2
    exit 1
  fi
done

STEP1_REF_USE=0
if [[ -n "${STEP1_REF_NPZ}" && -f "${STEP1_REF_NPZ}" ]]; then
  STEP1_REF_USE=1
elif [[ -n "${STEP1_REF_NPZ}" ]]; then
  echo "WARN: STEP1_REF_NPZ not found, running without external HLT override: ${STEP1_REF_NPZ}" >&2
fi

if [[ "${INCLUDE_HLT_CANDIDATE}" == "1" ]]; then
  case ",${CANDIDATE_MODELS_ALL}," in
    *,hlt,*) ;;
    *) CANDIDATE_MODELS_ALL="${CANDIDATE_MODELS_ALL},hlt" ;;
  esac
fi

export OUT_DIR FUSION_JSON
export M4_NPZ M9MID_NPZ M9HIGH_NPZ
export STEP1_REF_NPZ STEP1_REF_USE

python - <<'PY'
import json
from pathlib import Path
import os

out_dir = Path(os.environ["OUT_DIR"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
fusion_json = Path(os.environ["FUSION_JSON"]).resolve()

score_files = {
    "corrected_s01": str(Path(os.environ["M4_NPZ"]).resolve()),
    "offdrop_mid": str(Path(os.environ["M9MID_NPZ"]).resolve()),
    "offdrop_high": str(Path(os.environ["M9HIGH_NPZ"]).resolve()),
}

if str(os.environ.get("STEP1_REF_USE", "0")) == "1":
    step1_npz = str(Path(os.environ["STEP1_REF_NPZ"]).resolve())
    score_files["hlt"] = step1_npz
    score_files["teacher"] = step1_npz

fusion = {"run_dirs": {"score_files": score_files}}
fusion_json.write_text(json.dumps(fusion, indent=2), encoding="utf-8")
print(f"[prep] wrote fusion json: {fusion_json}")
PY

CMD=(
  python analyze_hlt_joint31_bin_gated_fusion.py
  --fusion_json "${FUSION_JSON}"
  --target_tprs "${TARGET_TPRS}"
  --anchor_model "${ANCHOR_MODEL}"
  --selection_mode "${SELECTION_MODE}"
  --candidate_models_all "${CANDIDATE_MODELS_ALL}"
  --router_cal_frac "${ROUTER_CAL_FRAC}"
  --seed "${SEED}"
  --calibration "${CALIBRATION}"
  --head_select_mode "${HEAD_SELECT_MODE}"
  --head_select_tpr "${HEAD_SELECT_TPR}"
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

echo "============================================================"
echo "3-Model Bin-Gated Fusion (m4 + m9mid + m9high + HLT)"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "TPRs:        ${TARGET_TPRS}"
echo "Selection:   ${SELECTION_MODE}"
echo "Calibration: ${CALIBRATION}"
echo "Head select: ${HEAD_SELECT_MODE} @TPR=${HEAD_SELECT_TPR}"
echo "Out dir:     ${OUT_DIR}"
echo "Fusion json: ${FUSION_JSON}"
echo "Models:      ${CANDIDATE_MODELS_ALL}"
if [[ "${STEP1_REF_USE}" == "1" ]]; then
  echo "Step1 ref:   ${STEP1_REF_NPZ}"
fi
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
