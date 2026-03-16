#!/usr/bin/env bash
# Fast Stage-C-only finetune from saved Stage2 checkpoint.
# Baseline "normal" Stage-C settings with lambda_cons=0.06.
#
# Defaults target your 100k/80 lcons0 run on research compute:
#   checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_lcons0
#
#SBATCH --job-name=stgCbase
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=offline_reconstructor_logs/stagec_from_stage2_baseline_%j.out
#SBATCH --error=offline_reconstructor_logs/stagec_from_stage2_baseline_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_lcons0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine}"
RUN_NAME="${RUN_NAME:-stagec_refine_baseline_lcons006}"

N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

# Baseline Stage-C settings ("normal" for this setup)
STAGEC_EPOCHS="${STAGEC_EPOCHS:-100}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-2e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-1e-5}"
LAMBDA_RECO="${LAMBDA_RECO:-0.35}"
LAMBDA_CONS="${LAMBDA_CONS:-0.02}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running fast Stage-C baseline from Stage2 checkpoint..."
echo "python finetune_stagec_from_stage2.py --run_dir ${RUN_DIR} --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --seed ${SEED} --stageC_epochs ${STAGEC_EPOCHS} --stageC_patience ${STAGEC_PATIENCE} --stageC_min_epochs ${STAGEC_MIN_EPOCHS} --stageC_lr_dual ${STAGEC_LR_DUAL} --stageC_lr_reco ${STAGEC_LR_RECO} --lambda_reco ${LAMBDA_RECO} --lambda_cons ${LAMBDA_CONS} --selection_metric ${SELECTION_METRIC} --device cuda"

python finetune_stagec_from_stage2.py \
  --run_dir "${RUN_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
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
  --lambda_reco "${LAMBDA_RECO}" \
  --lambda_cons "${LAMBDA_CONS}" \
  --selection_metric "${SELECTION_METRIC}" \
  --device cuda
