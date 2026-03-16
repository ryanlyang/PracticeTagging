#!/usr/bin/env bash
# Stage-C sweep suite debug_2 (lambda_reco down sweep).
#
# Submit:
#   sbatch sbatch/stagec_from_stage2_sweeps/run_stagec_sweep_debug_2.sh
#
#SBATCH --job-name=stgSwD2
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=offline_reconstructor_logs/stagec_sweep_debug2_%j.out
#SBATCH --error=offline_reconstructor_logs/stagec_sweep_debug2_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_lcons0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_sweeps}"

N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-100}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-2e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-1e-5}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

python sweep_stagec_from_stage2.py \
  --run_dir "${RUN_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --suite debug_2 \
  --run_prefix stagec_sweep \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --stageC_epochs "${STAGEC_EPOCHS}" \
  --stageC_patience "${STAGEC_PATIENCE}" \
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}" \
  --stageC_lr_dual "${STAGEC_LR_DUAL}" \
  --stageC_lr_reco "${STAGEC_LR_RECO}" \
  --selection_metric "${SELECTION_METRIC}" \
  --skip_existing \
  --device cuda
