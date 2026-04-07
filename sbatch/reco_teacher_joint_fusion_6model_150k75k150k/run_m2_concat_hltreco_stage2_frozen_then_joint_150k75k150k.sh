#!/usr/bin/env bash
#SBATCH --job-name=m2cncat
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_concat_hltreco_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_concat_hltreco_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_SUBPATH_DEFAULT="reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_hungarian_nophys_noratio/model2_joint_hungarian_nophys_noratio_150k75k150k_seed0"
RUN_DIR_DEFAULT_CHECKPOINTS="checkpoints/${RUN_SUBPATH_DEFAULT}"
RUN_DIR_DEFAULT_DOWNLOAD="download_checkpoints/${RUN_SUBPATH_DEFAULT}"

if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -d "${RUN_DIR_DEFAULT_CHECKPOINTS}" ]]; then
    RUN_DIR="${RUN_DIR_DEFAULT_CHECKPOINTS}"
  elif [[ -d "${RUN_DIR_DEFAULT_DOWNLOAD}" ]]; then
    RUN_DIR="${RUN_DIR_DEFAULT_DOWNLOAD}"
  else
    RUN_DIR="${RUN_DIR_DEFAULT_CHECKPOINTS}"
  fi
fi

TRAIN_PATH="${TRAIN_PATH:-./data}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_concat_hltreco_stage2}"
RUN_NAME="${RUN_NAME:-model2_concat_hltreco_stage2_150k75k150k_seed0}"

RECO_CKPT="${RECO_CKPT:-}"
BASELINE_CKPT="${BASELINE_CKPT:-}"

SEED="${SEED:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-6}"
DEVICE="${DEVICE:-cuda}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
USE_CORRECTED_FLAGS="${USE_CORRECTED_FLAGS:-0}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-45}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-10}"
JOINT_EPOCHS="${JOINT_EPOCHS:-25}"
JOINT_PATIENCE="${JOINT_PATIENCE:-8}"
JOINT_MIN_EPOCHS="${JOINT_MIN_EPOCHS:-8}"
JOINT_LR_MODEL="${JOINT_LR_MODEL:-2e-4}"
JOINT_LR_RECO="${JOINT_LR_RECO:-1e-4}"
JOINT_WEIGHT_DECAY="${JOINT_WEIGHT_DECAY:-1e-4}"
JOINT_WARMUP_EPOCHS="${JOINT_WARMUP_EPOCHS:-3}"
JOINT_LAMBDA_RECO="${JOINT_LAMBDA_RECO:-0.4}"
JOINT_LAMBDA_RANK="${JOINT_LAMBDA_RANK:-0.0}"
JOINT_LAMBDA_CONS="${JOINT_LAMBDA_CONS:-0.06}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

SAVE_FUSION_SCORES="${SAVE_FUSION_SCORES:-1}"
SKIP_SAVE_MODELS="${SKIP_SAVE_MODELS:-0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 1
fi
for f in data_setup.json data_splits.npz offline_reconstructor_stage2.pt baseline.pt; do
  if [[ ! -f "${RUN_DIR}/${f}" ]]; then
    echo "ERROR: Missing ${RUN_DIR}/${f}" >&2
    exit 1
  fi
done

CMD=(
  python train_m2_concat_hltreco_stage2_frozen_then_joint.py
  --run_dir "${RUN_DIR}"
  --train_path "${TRAIN_PATH}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --max_constits "${MAX_CONSTITS}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --pretrain_epochs "${PRETRAIN_EPOCHS}"
  --pretrain_patience "${PRETRAIN_PATIENCE}"
  --joint_epochs "${JOINT_EPOCHS}"
  --joint_patience "${JOINT_PATIENCE}"
  --joint_min_epochs "${JOINT_MIN_EPOCHS}"
  --joint_lr_model "${JOINT_LR_MODEL}"
  --joint_lr_reco "${JOINT_LR_RECO}"
  --joint_weight_decay "${JOINT_WEIGHT_DECAY}"
  --joint_warmup_epochs "${JOINT_WARMUP_EPOCHS}"
  --joint_lambda_reco "${JOINT_LAMBDA_RECO}"
  --joint_lambda_rank "${JOINT_LAMBDA_RANK}"
  --joint_lambda_cons "${JOINT_LAMBDA_CONS}"
  --selection_metric "${SELECTION_METRIC}"
)

if [[ -n "${RECO_CKPT}" ]]; then
  CMD+=(--reco_ckpt "${RECO_CKPT}")
fi
if [[ -n "${BASELINE_CKPT}" ]]; then
  CMD+=(--baseline_ckpt "${BASELINE_CKPT}")
fi
if [[ "${USE_CORRECTED_FLAGS}" == "1" ]]; then
  CMD+=(--use_corrected_flags)
fi
if [[ "${SAVE_FUSION_SCORES}" == "1" ]]; then
  CMD+=(--save_fusion_scores)
fi
if [[ "${SKIP_SAVE_MODELS}" == "1" ]]; then
  CMD+=(--skip_save_models)
fi

echo "============================================================"
echo "M2 Concat(HLT + Reco) from Stage2 Reconstructor"
echo "Source run: ${RUN_DIR}"
echo "Save dir:   ${SAVE_DIR}/${RUN_NAME}"
echo "Pretrain/Joint epochs: ${PRETRAIN_EPOCHS}/${JOINT_EPOCHS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Done: ${SAVE_DIR}/${RUN_NAME}"

