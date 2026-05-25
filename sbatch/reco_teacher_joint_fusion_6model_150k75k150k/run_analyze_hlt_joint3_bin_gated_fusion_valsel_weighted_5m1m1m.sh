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
import numpy as np

out_dir = Path(os.environ["OUT_DIR"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
fusion_json = Path(os.environ["FUSION_JSON"]).resolve()
m4_npz = Path(os.environ["M4_NPZ"]).resolve()
m9mid_npz = Path(os.environ["M9MID_NPZ"]).resolve()
m9high_npz = Path(os.environ["M9HIGH_NPZ"]).resolve()


def _first_key(z: np.lib.npyio.NpzFile, keys):
    for k in keys:
        if k in z:
            return k
    raise KeyError(f"None of keys found: {keys}")


z_mid = np.load(m9mid_npz)
y_val = np.asarray(z_mid["labels_val"], dtype=np.float32)
y_test = np.asarray(z_mid["labels_test"], dtype=np.float32)

k_joint_val = _first_key(
    z_mid,
    ["preds_residual_joint_val", "preds_residual_frozen_val", "preds_reco_teacher_val", "preds_hlt_val"],
)
k_joint_test = _first_key(
    z_mid,
    ["preds_residual_joint_test", "preds_residual_frozen_test", "preds_reco_teacher_test", "preds_hlt_test"],
)
k_hlt_val_mid = _first_key(z_mid, ["preds_hlt_val"])
k_hlt_test_mid = _first_key(z_mid, ["preds_hlt_test"])

preds_joint_val = np.asarray(z_mid[k_joint_val], dtype=np.float64)
preds_joint_test = np.asarray(z_mid[k_joint_test], dtype=np.float64)
preds_hlt_val = np.asarray(z_mid[k_hlt_val_mid], dtype=np.float64)
preds_hlt_test = np.asarray(z_mid[k_hlt_test_mid], dtype=np.float64)

preds_teacher_val = None
preds_teacher_test = None
if "preds_teacher_val" in z_mid and "preds_teacher_test" in z_mid:
    preds_teacher_val = np.asarray(z_mid["preds_teacher_val"], dtype=np.float64)
    preds_teacher_test = np.asarray(z_mid["preds_teacher_test"], dtype=np.float64)

# Optional HLT/teacher override from STEP1 artifact (only if labels match).
if str(os.environ.get("STEP1_REF_USE", "0")) == "1":
    step1_npz = Path(os.environ["STEP1_REF_NPZ"]).resolve()
    z_ref = np.load(step1_npz)
    if "labels_val" in z_ref and "labels_test" in z_ref:
        yv_ref = np.asarray(z_ref["labels_val"], dtype=np.float32)
        yt_ref = np.asarray(z_ref["labels_test"], dtype=np.float32)
        if np.array_equal(y_val, yv_ref) and np.array_equal(y_test, yt_ref):
            if "preds_hlt_val" in z_ref and "preds_hlt_test" in z_ref:
                preds_hlt_val = np.asarray(z_ref["preds_hlt_val"], dtype=np.float64)
                preds_hlt_test = np.asarray(z_ref["preds_hlt_test"], dtype=np.float64)
            if "preds_teacher_val" in z_ref and "preds_teacher_test" in z_ref:
                preds_teacher_val = np.asarray(z_ref["preds_teacher_val"], dtype=np.float64)
                preds_teacher_test = np.asarray(z_ref["preds_teacher_test"], dtype=np.float64)
            print(f"[prep] using STEP1 HLT/teacher override from: {step1_npz}")
        else:
            print(f"[prep] STEP1 labels mismatch; keeping HLT/teacher from m9mid: {step1_npz}")

joint_compat_npz = out_dir / "joint_delta_compat_from_offdrop_mid.npz"
save_pack = {
    "labels_val": y_val.astype(np.float32),
    "labels_test": y_test.astype(np.float32),
    "preds_hlt_val": preds_hlt_val.astype(np.float64),
    "preds_hlt_test": preds_hlt_test.astype(np.float64),
    "preds_joint_val": preds_joint_val.astype(np.float64),
    "preds_joint_test": preds_joint_test.astype(np.float64),
}
if preds_teacher_val is not None and preds_teacher_test is not None:
    save_pack["preds_teacher_val"] = preds_teacher_val.astype(np.float64)
    save_pack["preds_teacher_test"] = preds_teacher_test.astype(np.float64)
np.savez_compressed(joint_compat_npz, **save_pack)
print(
    "[prep] wrote joint_delta compatibility npz: "
    f"{joint_compat_npz} (joint keys: val={k_joint_val}, test={k_joint_test})"
)

score_files = {
    # analyze_hlt_joint31_* requires this key as its base score source.
    "joint_delta": str(joint_compat_npz),
    "corrected_s01": str(m4_npz),
    "offdrop_mid": str(m9mid_npz),
    "offdrop_high": str(m9high_npz),
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
