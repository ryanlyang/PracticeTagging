#!/usr/bin/env bash
# Stage-B-only run (frozen Stage A) with corrected merge/eff flags enabled.
# Seed fixed to 0 for reproducibility.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_stagebonly_50k40_flags_seed0.sh
#
#SBATCH --job-name=stgBFlg0
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_stageb_flags0_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_stageb_flags0_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stageb_only_50k_40c_flags_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
N_TRAIN_JETS="${N_TRAIN_JETS:-50000}"
MAX_CONSTITS="${MAX_CONSTITS:-40}"
OFFSET_JETS="${OFFSET_JETS:-0}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

STAGEB_EPOCHS="${STAGEB_EPOCHS:-45}"
STAGEB_PATIENCE="${STAGEB_PATIENCE:-12}"
STAGEB_MIN_EPOCHS="${STAGEB_MIN_EPOCHS:-12}"
STAGEB_LR_DUAL="${STAGEB_LR_DUAL:-4e-4}"
STAGEB_LAMBDA_RANK="${STAGEB_LAMBDA_RANK:-0.0}"
STAGEB_LAMBDA_CONS="${STAGEB_LAMBDA_CONS:-0.0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running Stage-B-only (flags ON) with seed=${SEED}"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --selection_metric auc \
  --stageB_epochs "${STAGEB_EPOCHS}" \
  --stageB_patience "${STAGEB_PATIENCE}" \
  --stageB_min_epochs "${STAGEB_MIN_EPOCHS}" \
  --stageB_lr_dual "${STAGEB_LR_DUAL}" \
  --stageB_lambda_rank "${STAGEB_LAMBDA_RANK}" \
  --stageB_lambda_cons "${STAGEB_LAMBDA_CONS}" \
  --use_corrected_flags \
  --stageC_epochs 0 \
  --stageC_min_epochs 0 \
  --disable_final_kd \
  --device cuda

