#!/usr/bin/env bash
#SBATCH --job-name=an55bgv
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze55_bingated_precomp_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze55_bingated_precomp_%j.err

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
CANDIDATE_MODELS_ALL="${CANDIDATE_MODELS_ALL:-}"
SELECTION_MODE="${SELECTION_MODE:-valsel}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.40}"
SEED="${SEED:-0}"
CALIBRATION="${CALIBRATION:-iso}"

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

DEV_POOL_SIZE="${DEV_POOL_SIZE:-1000000}"
DEV_POOL_OFFSET="${DEV_POOL_OFFSET:-0}"

PRECOMP_DIR="${PRECOMP_DIR:-}"
PRECOMP_SCORES_NPZ="${PRECOMP_SCORES_NPZ:-}"
PRECOMP_MANIFEST_JSON="${PRECOMP_MANIFEST_JSON:-}"

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

# Extend base 31-model fusion json to rolling 55-model set.
EXTRA_MODELS=(
  # Prior 15 additions
  "model2_joint_delta005_stagec_prog_unfreeze_splitctx_heads_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_jetlatent_set2set_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_unselected0_localr0_stagec_prog_wlocal010_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_hybridcand_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_splitcap120_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_linear_unsmear_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_hungarian_tokenaux_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_unselected0_localr0_stagec_prog_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_factorized_edit_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_eratioon_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_physon_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_splitmass_softmax_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_actionentropy_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_stagea_taskaux_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_splitk3_softgate_150k75k150k_seed0"
  # Additional 9 high-value additions
  "model2_joint_delta005_stagec_prog_unfreeze_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_ot_sinkhorn_nophys_noratio_150k75k150k_seed0"
  "model2_joint_hungarian_nophys_noratio_150k75k150k_seed0"
  "model2_joint_chamfer_phys0_eratio035_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_ot_chamfer_sinkhorn_nophys_noratio_150k75k150k_seed0"
  "model2_joint_sinkhorn_150k75k150k_seed0"
  "model2_joint_chamfer_budget_sparse_150k75k150k_seed0"
  "model2_joint_delta000_unsmearcap_trust_150k75k150k_seed0"
)

EXTENDED_FUSION_JSON="${EXTENDED_FUSION_JSON:-${FUSION_JSON%.json}_rolling55.json}"
export EXTRA_MODELS_CSV="$(IFS=,; echo "${EXTRA_MODELS[*]}")"
python3 - "${FUSION_JSON}" "${EXTENDED_FUSION_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
fusion = json.loads(src.read_text())

extra = [x.strip() for x in os.environ.get("EXTRA_MODELS_CSV", "").split(",") if x.strip()]
run_dirs = fusion.get("run_dirs", {})
if not isinstance(run_dirs, dict):
    run_dirs = {}
fusion["run_dirs"] = run_dirs
score_files = run_dirs.get("score_files", {})
if not isinstance(score_files, dict):
    score_files = {}
run_dirs["score_files"] = score_files

models_order = fusion.get("models_order", [])
if not isinstance(models_order, list):
    models_order = []

base_ckpt = "checkpoints/reco_teacher_joint_fusion_6model_150k75k150k"
suffix = "_150k75k150k_seed0"
for mid in extra:
    parent = mid[: -len(suffix)] if mid.endswith(suffix) else mid
    score_path = str((Path(base_ckpt) / parent / mid / "fusion_scores_val_test.npz").resolve())
    score_files[mid] = score_path
    if mid not in models_order:
        models_order.append(mid)

fusion["models_order"] = models_order
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(fusion, indent=2))
print(f"Wrote extended fusion json: {dst}")
print(f"Total models_order: {len(models_order)}")
PY

if [[ -z "${PRECOMP_DIR}" ]]; then
  PRECOMP_DIR="$(dirname "${EXTENDED_FUSION_JSON}")/precomputed_devpool_1000000"
fi
if [[ -z "${PRECOMP_SCORES_NPZ}" ]]; then
  PRECOMP_SCORES_NPZ="${PRECOMP_DIR}/precomputed_scores_devpool.npz"
fi
if [[ -z "${PRECOMP_MANIFEST_JSON}" ]]; then
  PRECOMP_MANIFEST_JSON="${PRECOMP_DIR}/precomputed_scores_manifest.json"
fi

if [[ ! -f "${PRECOMP_SCORES_NPZ}" ]]; then
  echo "ERROR: PRECOMP_SCORES_NPZ not found: ${PRECOMP_SCORES_NPZ}" >&2
  exit 2
fi
if [[ ! -f "${PRECOMP_MANIFEST_JSON}" ]]; then
  echo "ERROR: PRECOMP_MANIFEST_JSON not found: ${PRECOMP_MANIFEST_JSON}" >&2
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$(dirname "${EXTENDED_FUSION_JSON}")/bin_gated_fusion_55_valsel_1m"
fi

CMD=(
  python -u analyze_hlt_joint55_bin_gated_precomputed.py
  --precomputed_scores_npz "${PRECOMP_SCORES_NPZ}"
  --precomputed_manifest_json "${PRECOMP_MANIFEST_JSON}"
  --fusion_json "${EXTENDED_FUSION_JSON}"
  --target_tprs "${TARGET_TPRS}"
  --anchor_model "${ANCHOR_MODEL}"
  --selection_mode "${SELECTION_MODE}"
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
  --dev_pool_size "${DEV_POOL_SIZE}"
  --dev_pool_offset "${DEV_POOL_OFFSET}"
  --out_dir "${OUT_DIR}"
)

if [[ -n "${CANDIDATE_MODELS_ALL}" ]]; then
  CMD+=(--candidate_models_all "${CANDIDATE_MODELS_ALL}")
fi
if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi

echo "============================================================"
echo "55-Model Bin-Gated Fusion (Val-Selected, Precomputed 1M Pool)"
echo "Extended fusion json:  ${EXTENDED_FUSION_JSON}"
echo "Precomputed npz:       ${PRECOMP_SCORES_NPZ}"
echo "Precomputed manifest:  ${PRECOMP_MANIFEST_JSON}"
echo "TPRs:                  ${TARGET_TPRS}"
echo "Anchor:                ${ANCHOR_MODEL}"
echo "Selection:             ${SELECTION_MODE} (router_cal_frac=${ROUTER_CAL_FRAC})"
echo "Calibration:           ${CALIBRATION}"
echo "Dev pool size/off:     ${DEV_POOL_SIZE}/${DEV_POOL_OFFSET}"
echo "Global/Bin max_add:    ${GLOBAL_MAX_ADD}/${BIN_MAX_ADD}"
echo "w_step:                ${W_STEP}"
echo "Bin edges:             ${SCORE_BAND_EDGES} with dist cuts (${DIST_NEAR_CUT},${DIST_MID_LOW},${DIST_MID_HIGH})"
echo "Out dir:               ${OUT_DIR}"
echo "Heartbeat:             every 5 minutes"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

start_ts="$(date +%s)"
heartbeat() {
  while true; do
    sleep 300
    now_ts="$(date +%s)"
    elapsed="$(( now_ts - start_ts ))"
    h="$(( elapsed / 3600 ))"
    m="$(( (elapsed % 3600) / 60 ))"
    s="$(( elapsed % 60 ))"
    printf '[heartbeat] %s elapsed=%02d:%02d:%02d still running\n' "$(date '+%F %T')" "${h}" "${m}" "${s}"
  done
}
heartbeat &
hb_pid=$!
cleanup() {
  kill "${hb_pid}" 2>/dev/null || true
}
trap cleanup EXIT

"${CMD[@]}"

