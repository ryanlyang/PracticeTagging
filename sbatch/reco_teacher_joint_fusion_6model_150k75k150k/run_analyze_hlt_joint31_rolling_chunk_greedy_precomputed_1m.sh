#!/usr/bin/env bash
#SBATCH --job-name=an31rollpc
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_roll_precomp_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31_roll_precomp_%j.err

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

TARGET_TPR="${TARGET_TPR:-0.50}"
ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta}"
CANDIDATE_MODELS="${CANDIDATE_MODELS:-}"

VAL_POOL_SIZE="${VAL_POOL_SIZE:-1000000}"
VAL_POOL_OFFSET="${VAL_POOL_OFFSET:-0}"
N_CHUNKS="${N_CHUNKS:-5}"

CANDIDATE_TOPK_FIT="${CANDIDATE_TOPK_FIT:-31}"
MAX_STEPS="${MAX_STEPS:-20}"
PATIENCE="${PATIENCE:-5}"
W_STEP="${W_STEP:-0.01}"
CALIBRATION="${CALIBRATION:-raw}"
MIN_FIT_GAIN="${MIN_FIT_GAIN:-1e-6}"
MIN_CAL_GAIN="${MIN_CAL_GAIN:-1e-6}"
MIN_CUM_CAL_GAIN="${MIN_CUM_CAL_GAIN:-1e-6}"
TOPK_DIAGNOSTIC="${TOPK_DIAGNOSTIC:-10}"
SEED="${SEED:-0}"

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

if [[ -z "${PRECOMP_DIR}" ]]; then
  PRECOMP_DIR="$(dirname "${FUSION_JSON}")/precomputed_devpool_1000000"
fi
if [[ -z "${PRECOMP_SCORES_NPZ}" ]]; then
  PRECOMP_SCORES_NPZ="${PRECOMP_DIR}/precomputed_scores_devpool.npz"
fi
if [[ -z "${PRECOMP_MANIFEST_JSON}" ]]; then
  PRECOMP_MANIFEST_JSON="${PRECOMP_DIR}/precomputed_scores_manifest.json"
fi

if [[ ! -f "${PRECOMP_SCORES_NPZ}" ]]; then
  echo "ERROR: PRECOMP_SCORES_NPZ not found: ${PRECOMP_SCORES_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${PRECOMP_MANIFEST_JSON}" ]]; then
  echo "ERROR: PRECOMP_MANIFEST_JSON not found: ${PRECOMP_MANIFEST_JSON}" >&2
  exit 1
fi

CMD=(
  python analyze_hlt_joint31_rolling_chunk_greedy.py
  --fusion_json "${FUSION_JSON}"
  --precomputed_scores_npz "${PRECOMP_SCORES_NPZ}"
  --precomputed_manifest_json "${PRECOMP_MANIFEST_JSON}"
  --target_tpr "${TARGET_TPR}"
  --anchor_model "${ANCHOR_MODEL}"
  --candidate_topk_fit "${CANDIDATE_TOPK_FIT}"
  --val_pool_size "${VAL_POOL_SIZE}"
  --val_pool_offset "${VAL_POOL_OFFSET}"
  --n_chunks "${N_CHUNKS}"
  --max_steps "${MAX_STEPS}"
  --patience "${PATIENCE}"
  --w_step "${W_STEP}"
  --calibration "${CALIBRATION}"
  --min_fit_gain "${MIN_FIT_GAIN}"
  --min_cal_gain "${MIN_CAL_GAIN}"
  --min_cum_cal_gain "${MIN_CUM_CAL_GAIN}"
  --topk_diagnostic "${TOPK_DIAGNOSTIC}"
  --seed "${SEED}"
)

if [[ -n "${CANDIDATE_MODELS}" ]]; then
  CMD+=(--candidate_models "${CANDIDATE_MODELS}")
fi
if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi

echo "============================================================"
echo "31-Model Rolling-Chunk Greedy Fusion (Precomputed Dev Pool)"
echo "Fusion json:         ${FUSION_JSON}"
echo "Precomputed npz:     ${PRECOMP_SCORES_NPZ}"
echo "Precomputed manifest:${PRECOMP_MANIFEST_JSON}"
echo "Target TPR:          ${TARGET_TPR}"
echo "Anchor:              ${ANCHOR_MODEL}"
echo "Val pool size/offset:${VAL_POOL_SIZE}/${VAL_POOL_OFFSET}"
echo "Chunks:              ${N_CHUNKS}"
echo "Greedy:              topk=${CANDIDATE_TOPK_FIT}, max_steps=${MAX_STEPS}, w_step=${W_STEP}"
echo "Acceptance gains:    fit>=${MIN_FIT_GAIN}, cal>=${MIN_CAL_GAIN}, cum>=${MIN_CUM_CAL_GAIN}"
echo "Calibration:         ${CALIBRATION}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
