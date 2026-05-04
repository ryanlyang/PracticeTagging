#!/usr/bin/env bash
#SBATCH --job-name=an5bgw
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze5_bingated_valsel_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze5_bingated_valsel_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

M15_RUN_DIR="${M15_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model15_dualreco_dualview_offdrop_mid_weighted_150k150k300k/model15_dualreco_dualview_offdrop_mid_weighted_150k150k300k_seed0}"
M16_RUN_DIR="${M16_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model16_dualreco_dualview_topk60_weighted_150k150k300k/model16_dualreco_dualview_topk60_weighted_150k150k300k_seed0}"
M17_RUN_DIR="${M17_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model17_dualreco_dualview_antioverlap_weighted_150k150k300k/model17_dualreco_dualview_antioverlap_weighted_150k150k300k_seed0}"
M6_RUN_DIR="${M6_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model6_concat_stagea_corrected_weighted_150k150k300k/model6_concat_stagea_corrected_weighted_150k150k300k_seed0}"
M9_RUN_DIR="${M9_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_high_weighted_150k150k300k/model9_stageA_residual_hlt_offdrop_high_weighted_150k150k300k_seed0}"

M15_NPZ="${M15_NPZ:-${M15_RUN_DIR}/dualreco_dualview_scores.npz}"
M16_NPZ="${M16_NPZ:-${M16_RUN_DIR}/dualreco_dualview_scores.npz}"
M17_NPZ="${M17_NPZ:-${M17_RUN_DIR}/dualreco_dualview_scores.npz}"
M6_NPZ="${M6_NPZ:-${M6_RUN_DIR}/concat_teacher_stageA_scores.npz}"
M9_NPZ="${M9_NPZ:-${M9_RUN_DIR}/stageA_residual_scores.npz}"

ANCHOR_MODEL="${ANCHOR_MODEL:-dual_m15_offdrop_mid}"
TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
CALIBRATION="${CALIBRATION:-iso}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.40}"
SEED="${SEED:-0}"

# Finer search than the default 31-model script.
GLOBAL_MAX_ADD="${GLOBAL_MAX_ADD:-4}"
BIN_MAX_ADD="${BIN_MAX_ADD:-4}"
W_STEP="${W_STEP:-0.0025}"
MIN_BIN_FIT="${MIN_BIN_FIT:-2000}"
MIN_GLOBAL_IMPROVE="${MIN_GLOBAL_IMPROVE:-2e-7}"
MIN_BIN_IMPROVE="${MIN_BIN_IMPROVE:-1e-6}"

SCORE_BAND_EDGES="${SCORE_BAND_EDGES:-0.0,0.8,0.9,1.0}"
DIST_NEAR_CUT="${DIST_NEAR_CUT:-0.0384}"
DIST_MID_LOW="${DIST_MID_LOW:-0.06285}"
DIST_MID_HIGH="${DIST_MID_HIGH:-0.07386}"

OUT_DIR="${OUT_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/bin_gated_fusion_5_weighted_150k150k300k_valsel}"
FUSION_JSON="${FUSION_JSON:-${OUT_DIR}/fusion_hlt_joint5_weighted_150k150k300k.json}"
BOOTSTRAP_JOINT_NPZ="${BOOTSTRAP_JOINT_NPZ:-${OUT_DIR}/joint_bootstrap_scores_from_m15.npz}"

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

for p in "${M15_NPZ}" "${M16_NPZ}" "${M17_NPZ}" "${M6_NPZ}" "${M9_NPZ}"; do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: missing required score file: ${p}" >&2
    exit 1
  fi
done

# Build a minimal "joint_delta-compatible" bootstrap npz + fusion json
# so analyze_hlt_joint31_bin_gated_fusion.py can be reused for this 5-model setup.
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import os

out_dir = Path(os.environ["OUT_DIR"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

m15_npz = Path(os.environ["M15_NPZ"]).resolve()
m16_npz = Path(os.environ["M16_NPZ"]).resolve()
m17_npz = Path(os.environ["M17_NPZ"]).resolve()
m6_npz = Path(os.environ["M6_NPZ"]).resolve()
m9_npz = Path(os.environ["M9_NPZ"]).resolve()

bootstrap_npz = Path(os.environ["BOOTSTRAP_JOINT_NPZ"]).resolve()
fusion_json = Path(os.environ["FUSION_JSON"]).resolve()

z = np.load(m15_npz)
labels_val = np.asarray(z["labels_val"], dtype=np.float32)
labels_test = np.asarray(z["labels_test"], dtype=np.float32)
preds_hlt_val = np.asarray(z["preds_hlt_val"], dtype=np.float64)
preds_hlt_test = np.asarray(z["preds_hlt_test"], dtype=np.float64)
preds_teacher_val = np.asarray(z["preds_teacher_val"], dtype=np.float64) if "preds_teacher_val" in z else preds_hlt_val
preds_teacher_test = np.asarray(z["preds_teacher_test"], dtype=np.float64) if "preds_teacher_test" in z else preds_hlt_test

# "preds_joint_*" are required by the loader; use anchor-equivalent scores.
if "preds_dual_frozen_val" in z and "preds_dual_frozen_test" in z:
    preds_joint_val = np.asarray(z["preds_dual_frozen_val"], dtype=np.float64)
    preds_joint_test = np.asarray(z["preds_dual_frozen_test"], dtype=np.float64)
elif "preds_dual_joint_val" in z and "preds_dual_joint_test" in z:
    preds_joint_val = np.asarray(z["preds_dual_joint_val"], dtype=np.float64)
    preds_joint_test = np.asarray(z["preds_dual_joint_test"], dtype=np.float64)
else:
    preds_joint_val = preds_hlt_val.copy()
    preds_joint_test = preds_hlt_test.copy()

np.savez_compressed(
    bootstrap_npz,
    labels_val=labels_val,
    labels_test=labels_test,
    preds_hlt_val=preds_hlt_val,
    preds_hlt_test=preds_hlt_test,
    preds_teacher_val=preds_teacher_val,
    preds_teacher_test=preds_teacher_test,
    preds_joint_val=preds_joint_val,
    preds_joint_test=preds_joint_test,
)

fusion = {
    "run_dirs": {
        "score_files": {
            "joint_delta": str(bootstrap_npz),
            "dual_m15_offdrop_mid": str(m15_npz),
            "dual_m16_topk60": str(m16_npz),
            "dual_m17_antioverlap": str(m17_npz),
            "concat_corrected": str(m6_npz),
            "offdrop_high": str(m9_npz),
        }
    }
}
fusion_json.write_text(json.dumps(fusion, indent=2), encoding="utf-8")
print(f"[prep] wrote bootstrap npz: {bootstrap_npz}")
print(f"[prep] wrote fusion json   : {fusion_json}")
PY

CMD=(
  python analyze_hlt_joint31_bin_gated_fusion.py
  --fusion_json "${FUSION_JSON}"
  --target_tprs "${TARGET_TPRS}"
  --anchor_model "${ANCHOR_MODEL}"
  --selection_mode "${SELECTION_MODE}"
  --candidate_models_all "dual_m15_offdrop_mid,concat_corrected,dual_m17_antioverlap,dual_m16_topk60,offdrop_high"
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
echo "5-Model Bin-Gated Fusion (Weighted 150k/150k/300k, Fine-Grained)"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "TPRs:        ${TARGET_TPRS}"
echo "Selection:   ${SELECTION_MODE}"
echo "Calibration: ${CALIBRATION}"
echo "Out dir:     ${OUT_DIR}"
echo "Fusion json: ${FUSION_JSON}"
echo "Models:      dual_m15_offdrop_mid, concat_corrected, dual_m17_antioverlap, dual_m16_topk60, offdrop_high"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
