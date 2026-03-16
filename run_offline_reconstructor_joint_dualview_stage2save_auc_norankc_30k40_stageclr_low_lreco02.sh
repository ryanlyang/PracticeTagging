#!/usr/bin/env bash
# 30k/40, low Stage-C LR + lambda_reco=0.2
#SBATCH --job-name=offreco30r2
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_30k40_lr_r02_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_30k40_lr_r02_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_30k_40c_auc_norankc_stageclr_low_lreco02}"
N_TRAIN_JETS="${N_TRAIN_JETS:-30000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-40}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
SEED_TAG="${SEED_TAG:-0}"

export PYTHONHASHSEED="${SEED_TAG}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running 30k/40 low Stage-C LR + lambda_reco=0.2"
echo "Seed tag: ${SEED_TAG} (script also uses fixed internal RANDOM_SEED)."

echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --lambda_cons 0.0 --lambda_reco 0.2 --stageC_lr_dual 2e-5 --stageC_lr_reco 1e-5 --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --lambda_cons 0.0 \
  --lambda_reco 0.1 \
  --stageC_lr_dual 2e-5 \
  --stageC_lr_reco 1e-5 \
  --disable_final_kd \
  --device cuda
