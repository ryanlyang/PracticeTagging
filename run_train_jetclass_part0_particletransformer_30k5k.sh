#!/usr/bin/env bash
#SBATCH --job-name=jcPart0PT
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_part0_pt_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_part0_pt_%j.err

set -euo pipefail

# Defaults tuned for your current setup; override with env vars if needed.
DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
JETCLASS_REPO="${JETCLASS_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_NAME="${RUN_NAME:-jetclass_part0_pt_30k5k}"
FEATURE_TYPE="${FEATURE_TYPE:-full}"

TRAIN_FILES_PER_CLASS="${TRAIN_FILES_PER_CLASS:-8}"
VAL_FILES_PER_CLASS="${VAL_FILES_PER_CLASS:-1}"
TEST_FILES_PER_CLASS="${TEST_FILES_PER_CLASS:-1}"
SEED="${SEED:-52}"

NUM_EPOCHS="${NUM_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
START_LR="${START_LR:-1e-3}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-30000}"
SAMPLES_PER_EPOCH_VAL="${SAMPLES_PER_EPOCH_VAL:-5000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
FETCH_STEP="${FETCH_STEP:-0.05}"
GPUS="${GPUS:-0}"

LOG_FILE="${LOG_FILE:-offline_reconstructor_logs/${RUN_NAME}.log}"
PREDICT_OUTPUT="${PREDICT_OUTPUT:-pred_${RUN_NAME}.root}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_part0_part}"
TENSORBOARD="${TENSORBOARD:-JetClass_part0_ParT}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

if ! command -v weaver >/dev/null 2>&1; then
  echo "[preflight] weaver executable not found in PATH; checking python module..."
  if ! python -m weaver --help >/dev/null 2>&1; then
    echo "[preflight] ERROR: neither 'weaver' executable nor python module 'weaver' is available in this env."
    echo "[preflight] Install with: python -m pip install --user weaver-core"
    exit 1
  fi
fi

cd "${SLURM_SUBMIT_DIR}"
mkdir -p offline_reconstructor_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

echo "=================================================="
echo "JetClass part0 ParticleTransformer run"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Python: $(which python)"
python - << 'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

echo ""
echo "Running command:"
echo "python train_jetclass_part0_particletransformer.py ..."

python train_jetclass_part0_particletransformer.py \
  --data_dir "${DATA_DIR}" \
  --jetclass_repo "${JETCLASS_REPO}" \
  --feature_type "${FEATURE_TYPE}" \
  --train_files_per_class "${TRAIN_FILES_PER_CLASS}" \
  --val_files_per_class "${VAL_FILES_PER_CLASS}" \
  --test_files_per_class "${TEST_FILES_PER_CLASS}" \
  --seed "${SEED}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --start_lr "${START_LR}" \
  --samples_per_epoch "${SAMPLES_PER_EPOCH}" \
  --samples_per_epoch_val "${SAMPLES_PER_EPOCH_VAL}" \
  --num_workers "${NUM_WORKERS}" \
  --fetch_step "${FETCH_STEP}" \
  --gpus "${GPUS}" \
  --save_root "${SAVE_ROOT}" \
  --log_file "${LOG_FILE}" \
  --predict_output "${PREDICT_OUTPUT}" \
  --tensorboard "${TENSORBOARD}" \
  --run_name "${RUN_NAME}" \
  --use_amp

