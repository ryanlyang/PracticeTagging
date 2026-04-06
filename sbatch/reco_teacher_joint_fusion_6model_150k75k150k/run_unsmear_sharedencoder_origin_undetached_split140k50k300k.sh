#!/usr/bin/env bash
#SBATCH --job-name=unsm14050300
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=offline_reconstructor_logs/unsmear_sharedencoder_joint_new/unsmear_sharedencoder_split140k50k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/unsmear_sharedencoder_joint_new/unsmear_sharedencoder_split140k50k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/unsmear_sharedencoder_joint_new

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local start="$1"
  local cur
  cur="$(cd "${start}" 2>/dev/null && pwd || true)"
  if [[ -z "${cur}" ]]; then
    return 1
  fi
  while [[ "${cur}" != "/" ]]; do
    if [[ -f "${cur}/TaggingNew/pure_unsmear/joint_new/model.py" ]]; then
      echo "${cur}"
      return 0
    fi
    cur="$(dirname "${cur}")"
  done
  return 1
}

if [[ -z "${REPO_ROOT:-}" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(pwd)" \
    "${SCRIPT_DIR}" \
    "${HOME}/atlas/PracticeTagging"
  do
    if [[ -n "${candidate}" ]]; then
      if REPO_FOUND="$(find_repo_root "${candidate}")"; then
        REPO_ROOT="${REPO_FOUND}"
        break
      fi
    fi
  done
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "ERROR: Could not locate repo root." >&2
  echo "Tried from: SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', pwd='$(pwd)', script_dir='${SCRIPT_DIR}'." >&2
  echo "Set REPO_ROOT explicitly, e.g. REPO_ROOT=/home/ryreu/atlas/PracticeTagging sbatch <script>." >&2
  exit 2
fi

PY_SCRIPT_DEFAULT="${REPO_ROOT}/TaggingNew/pure_unsmear/joint_new/run_unsmear_sharedencoder_delta_gate_split140k50k300k.py"
PY_SCRIPT="${PY_SCRIPT:-${PY_SCRIPT_DEFAULT}}"

if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -f "${REPO_ROOT}/data/test.h5" ]]; then
    DATA_PATH="${REPO_ROOT}/data/test.h5"
  else
    DATA_PATH="${REPO_ROOT}/test.h5"
  fi
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/TaggingNew/pure_unsmear/joint_new/runs}"
SHARED_BASELINE_DIR="${SHARED_BASELINE_DIR:-${REPO_ROOT}/TaggingNew/pure_unsmear/joint_new/runs/shared_offline_hlt_baselines}"
RUN_NAME="${RUN_NAME:-unsmear_transformer_sharedencoder_delta_gate_joint_split140k50k300k_seed42}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

TRAIN_COUNT="${TRAIN_COUNT:-140000}"
VAL_COUNT="${VAL_COUNT:-50000}"
TEST_COUNT="${TEST_COUNT:-300000}"

BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
PATIENCE="${PATIENCE:-8}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_auc_weighted}"

KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
KD_ALPHA="${KD_ALPHA:-0.5}"
KD_ALPHA_ATTN="${KD_ALPHA_ATTN:-0.0}"

JOINT_UNSMEAR_WEIGHT="${JOINT_UNSMEAR_WEIGHT:-1.6}"
JOINT_CLS_WEIGHT="${JOINT_CLS_WEIGHT:-0.8}"
JOINT_PHYS_WEIGHT="${JOINT_PHYS_WEIGHT:-0.0}"
JOINT_UNSMEAR_LOSS_MODE="${JOINT_UNSMEAR_LOSS_MODE:-mask}"

CLS_USE_DELTA_FUSION="${CLS_USE_DELTA_FUSION:-true}"
CLS_DETACH_DELTA_FOR_CLS="${CLS_DETACH_DELTA_FOR_CLS:-false}"
CLS_GATE_HIDDEN_DIM="${CLS_GATE_HIDDEN_DIM:-128}"
CLS_GATE_INIT_BIAS="${CLS_GATE_INIT_BIAS:--2.0}"
CLS_ALPHA_INIT="${CLS_ALPHA_INIT:-0.05}"

LOAD_SHARED_BASELINES="${LOAD_SHARED_BASELINES:-false}"
LOAD_JOINT_MODEL="${LOAD_JOINT_MODEL:-false}"
RESMEAR_EACH_EPOCH_BASELINES="${RESMEAR_EACH_EPOCH_BASELINES:-true}"
RESMEAR_EACH_EPOCH_JOINT="${RESMEAR_EACH_EPOCH_JOINT:-true}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${REPO_ROOT}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: Python runner not found: ${PY_SCRIPT}" >&2
  echo "Expected file in this checkout:" >&2
  echo "  ${PY_SCRIPT_DEFAULT}" >&2
  echo "Current repo root: ${REPO_ROOT}" >&2
  echo "If needed, set PY_SCRIPT=/abs/path/to/run_unsmear_sharedencoder_delta_gate_split140k50k300k.py" >&2
  exit 2
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH does not exist: ${DATA_PATH}" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"

CMD=(
  python "${PY_SCRIPT}"
  --data_path "${DATA_PATH}"
  --run_name "${RUN_NAME}"
  --output_root "${OUTPUT_ROOT}"
  --shared_baseline_dir "${SHARED_BASELINE_DIR}"
  --seed "${SEED}"
  --max_particles 100
  --feature_kind 7d
  --train_count "${TRAIN_COUNT}"
  --val_count "${VAL_COUNT}"
  --test_count "${TEST_COUNT}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --patience "${PATIENCE}"
  --early_stop_metric "${EARLY_STOP_METRIC}"
  --load_shared_baselines "${LOAD_SHARED_BASELINES}"
  --load_joint_model "${LOAD_JOINT_MODEL}"
  --resmear_each_epoch_baselines "${RESMEAR_EACH_EPOCH_BASELINES}"
  --resmear_each_epoch_joint "${RESMEAR_EACH_EPOCH_JOINT}"
  --kd_temperature "${KD_TEMPERATURE}"
  --kd_alpha "${KD_ALPHA}"
  --kd_alpha_attn "${KD_ALPHA_ATTN}"
  --joint_unsmear_weight "${JOINT_UNSMEAR_WEIGHT}"
  --joint_cls_weight "${JOINT_CLS_WEIGHT}"
  --joint_phys_weight "${JOINT_PHYS_WEIGHT}"
  --joint_unsmear_loss_mode "${JOINT_UNSMEAR_LOSS_MODE}"
  --cls_use_delta_fusion "${CLS_USE_DELTA_FUSION}"
  --cls_detach_delta_for_cls "${CLS_DETACH_DELTA_FOR_CLS}"
  --cls_gate_hidden_dim "${CLS_GATE_HIDDEN_DIM}"
  --cls_gate_init_bias "${CLS_GATE_INIT_BIAS}"
  --cls_alpha_init "${CLS_ALPHA_INIT}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)

echo "============================================================"
echo "SharedEncoder Unsmear Repro (undetached delta fusion)"
echo "Repo root: ${REPO_ROOT}"
echo "Data path: ${DATA_PATH}"
echo "Run: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "Split counts: train=${TRAIN_COUNT}, val=${VAL_COUNT}, test=${TEST_COUNT}"
echo "Joint unsmear loss mode: ${JOINT_UNSMEAR_LOSS_MODE}"
echo "Seed: ${SEED}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${OUTPUT_ROOT}/${RUN_NAME}"
