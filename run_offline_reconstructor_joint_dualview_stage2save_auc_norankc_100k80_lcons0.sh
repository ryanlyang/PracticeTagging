#!/usr/bin/env bash
# 100k/80 run using offline_reconstructor_joint_dualview_stage2save_auc_norankc.py
# - Top taggers selected by val AUC
# - Stage C rank term disabled
# - Jet regressor OFF
# - Stage C lambda_cons forced to 0
#SBATCH --job-name=offrecoA0C
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_auc100c0_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_auc100c0_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stage2save_auc_norankc_lcons0}"
N_TRAIN_JETS="${N_TRAIN_JETS:-30000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-40}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running 100k/80 AUC-select no-rankC config (jet reg OFF, lambda_cons=0):"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --lambda_cons 0.0 --disable_final_kd --device cuda"

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
  --disable_final_kd \
  --device cuda
