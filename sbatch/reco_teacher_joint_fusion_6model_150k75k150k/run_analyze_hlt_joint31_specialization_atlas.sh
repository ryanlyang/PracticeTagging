#!/usr/bin/env bash
#SBATCH --job-name=an31atlas
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_atlas_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_atlas_%j.err

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

TRAIN_PATH="${TRAIN_PATH:-./data}"
TARGET_TPRS="${TARGET_TPRS:-0.50,0.30}"
ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta}"
QUANTILE_BINS="${QUANTILE_BINS:-10}"
COUNT_EDGES="${COUNT_EDGES:-0,10,20,30,40,60,80,100,200}"
MIN_BIN_NEG="${MIN_BIN_NEG:-200}"

GREEDY_MAX_ADD="${GREEDY_MAX_ADD:-12}"
GREEDY_W_STEP="${GREEDY_W_STEP:-0.01}"
GREEDY_CALIBRATION="${GREEDY_CALIBRATION:-iso}"

HLT_SEED="${HLT_SEED:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

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

CMD=(
  python analyze_hlt_joint31_specialization_atlas.py
  --fusion_json "${FUSION_JSON}"
  --train_path "${TRAIN_PATH}"
  --target_tprs "${TARGET_TPRS}"
  --anchor_model "${ANCHOR_MODEL}"
  --quantile_bins "${QUANTILE_BINS}"
  --count_edges "${COUNT_EDGES}"
  --min_bin_neg "${MIN_BIN_NEG}"
  --greedy_max_add "${GREEDY_MAX_ADD}"
  --greedy_w_step "${GREEDY_W_STEP}"
  --greedy_calibration "${GREEDY_CALIBRATION}"
  --hlt_seed "${HLT_SEED}"
  --max_constits "${MAX_CONSTITS}"
)

if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi

echo "============================================================"
echo "31-Model Specialization Atlas"
echo "Fusion json: ${FUSION_JSON}"
echo "Train path:  ${TRAIN_PATH}"
echo "TPRs:        ${TARGET_TPRS}"
echo "Anchor:      ${ANCHOR_MODEL}"
echo "Greedy:      max_add=${GREEDY_MAX_ADD}, w_step=${GREEDY_W_STEP}, cal=${GREEDY_CALIBRATION}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

