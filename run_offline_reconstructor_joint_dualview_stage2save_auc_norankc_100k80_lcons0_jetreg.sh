#!/usr/bin/env bash
# 100k/80 run using offline_reconstructor_joint_dualview_stage2save_auc_norankc.py
# - Top taggers selected by val AUC
# - Stage C rank term disabled
# - Jet regressor ON
# - Stage C lambda_cons forced to 0
#SBATCH --job-name=offrecoA0J
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_auc100c0jr_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_auc100c0jr_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stage2save_auc_norankc_lcons0_jetreg}"
N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running 100k/80 AUC-select no-rankC config (jet reg ON, lambda_cons=0):"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --lambda_cons 0.0 --enable_jet_regressor --disable_final_kd --device cuda"

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
  --enable_jet_regressor \
  --disable_final_kd \
  --device cuda
