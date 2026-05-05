#!/usr/bin/env bash
#SBATCH --job-name=an12bgw
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze12_bingated_valsel_weighted_500k300k500k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze12_bingated_valsel_weighted_500k300k500k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

M2D005_RUN_DIR="${M2D005_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_500k300k500k/model2_joint_delta005_weighted_500k300k500k_seed0}"
M2D020_RUN_DIR="${M2D020_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta020_weighted_500k300k500k/model2_joint_delta020_weighted_500k300k500k_seed0}"
M4_RUN_DIR="${M4_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model4_recoteacher_s01_corrected_weighted_500k300k500k/model4_recoteacher_s01_corrected_weighted_500k300k500k_seed0}"
M5_RUN_DIR="${M5_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model5_joint_s01_full_weighted_500k300k500k/model5_joint_s01_full_weighted_500k300k500k_seed0}"
M6_RUN_DIR="${M6_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model6_concat_stagea_corrected_weighted_500k300k500k/model6_concat_stagea_corrected_weighted_500k300k500k_seed0}"
M9MID_RUN_DIR="${M9MID_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_mid_weighted_500k300k500k/model9_stageA_residual_hlt_offdrop_mid_weighted_500k300k500k_seed0}"
M9HIGH_RUN_DIR="${M9HIGH_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_high_weighted_500k300k500k/model9_stageA_residual_hlt_offdrop_high_weighted_500k300k500k_seed0}"
M12_RUN_DIR="${M12_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model12_dualreco_dualview_feat_noscale_weighted_500k300k500k/model12_dualreco_dualview_feat_noscale_weighted_500k300k500k_seed0}"
M15MID_RUN_DIR="${M15MID_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model15_dualreco_dualview_offdrop_mid_weighted_500k300k500k/model15_dualreco_dualview_offdrop_mid_weighted_500k300k500k_seed0}"
M15HIGH_RUN_DIR="${M15HIGH_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model15_dualreco_dualview_offdrop_high_weighted_500k300k500k/model15_dualreco_dualview_offdrop_high_weighted_500k300k500k_seed0}"
M16_RUN_DIR="${M16_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model16_dualreco_dualview_topk60_weighted_500k300k500k/model16_dualreco_dualview_topk60_weighted_500k300k500k_seed0}"
M17_RUN_DIR="${M17_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model17_dualreco_dualview_antioverlap_weighted_500k300k500k/model17_dualreco_dualview_antioverlap_weighted_500k300k500k_seed0}"

M2D005_NPZ="${M2D005_NPZ:-${M2D005_RUN_DIR}/fusion_scores_val_test.npz}"
M2D020_NPZ="${M2D020_NPZ:-${M2D020_RUN_DIR}/fusion_scores_val_test.npz}"
M4_NPZ="${M4_NPZ:-${M4_RUN_DIR}/stageA_only_scores.npz}"
M5_NPZ="${M5_NPZ:-${M5_RUN_DIR}/fusion_scores_val_test.npz}"
M6_NPZ="${M6_NPZ:-${M6_RUN_DIR}/concat_teacher_stageA_scores.npz}"
M9MID_NPZ="${M9MID_NPZ:-${M9MID_RUN_DIR}/stageA_residual_scores.npz}"
M9HIGH_NPZ="${M9HIGH_NPZ:-${M9HIGH_RUN_DIR}/stageA_residual_scores.npz}"
M12_NPZ="${M12_NPZ:-${M12_RUN_DIR}/dualreco_dualview_scores.npz}"
M15MID_NPZ="${M15MID_NPZ:-${M15MID_RUN_DIR}/dualreco_dualview_scores.npz}"
M15HIGH_NPZ="${M15HIGH_NPZ:-${M15HIGH_RUN_DIR}/dualreco_dualview_scores.npz}"
M16_NPZ="${M16_NPZ:-${M16_RUN_DIR}/dualreco_dualview_scores.npz}"
M17_NPZ="${M17_NPZ:-${M17_RUN_DIR}/dualreco_dualview_scores.npz}"

ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta}"
TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
CALIBRATION="${CALIBRATION:-iso}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.40}"
SEED="${SEED:-0}"

CANDIDATE_MODELS_ALL="${CANDIDATE_MODELS_ALL:-joint_delta,joint_delta020,joint_s01,corrected_s01,concat_corrected,offdrop_mid,offdrop_high,dual_m12_noscale,dual_m15_offdrop_mid,dual_m15_offdrop_high,dual_m16_topk60,dual_m17_antioverlap}"

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

OUT_DIR="${OUT_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_12_weighted_500k300k500k_valsel}"
FUSION_JSON="${FUSION_JSON:-${OUT_DIR}/fusion_hlt_joint12_weighted_500k300k500k.json}"

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

for p in \
  "${M2D005_NPZ}" "${M2D020_NPZ}" "${M4_NPZ}" "${M5_NPZ}" "${M6_NPZ}" \
  "${M9MID_NPZ}" "${M9HIGH_NPZ}" "${M12_NPZ}" "${M15MID_NPZ}" "${M15HIGH_NPZ}" "${M16_NPZ}" "${M17_NPZ}"
do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing required score file: ${p}" >&2
    exit 1
  fi
done

python - <<'PY'
import json
from pathlib import Path
import os

out_dir = Path(os.environ["OUT_DIR"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
fusion_json = Path(os.environ["FUSION_JSON"]).resolve()

score_files = {
    "joint_delta": str(Path(os.environ["M2D005_NPZ"]).resolve()),
    "joint_delta020": str(Path(os.environ["M2D020_NPZ"]).resolve()),
    "joint_s01": str(Path(os.environ["M5_NPZ"]).resolve()),
    "corrected_s01": str(Path(os.environ["M4_NPZ"]).resolve()),
    "concat_corrected": str(Path(os.environ["M6_NPZ"]).resolve()),
    "offdrop_mid": str(Path(os.environ["M9MID_NPZ"]).resolve()),
    "offdrop_high": str(Path(os.environ["M9HIGH_NPZ"]).resolve()),
    "dual_m12_noscale": str(Path(os.environ["M12_NPZ"]).resolve()),
    "dual_m15_offdrop_mid": str(Path(os.environ["M15MID_NPZ"]).resolve()),
    "dual_m15_offdrop_high": str(Path(os.environ["M15HIGH_NPZ"]).resolve()),
    "dual_m16_topk60": str(Path(os.environ["M16_NPZ"]).resolve()),
    "dual_m17_antioverlap": str(Path(os.environ["M17_NPZ"]).resolve()),
}

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
echo "12-Model Bin-Gated Fusion (Weighted 500k/300k/500k, Fine-Grained)"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "TPRs:        ${TARGET_TPRS}"
echo "Selection:   ${SELECTION_MODE}"
echo "Calibration: ${CALIBRATION}"
echo "Out dir:     ${OUT_DIR}"
echo "Fusion json: ${FUSION_JSON}"
echo "Models:      ${CANDIDATE_MODELS_ALL}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
