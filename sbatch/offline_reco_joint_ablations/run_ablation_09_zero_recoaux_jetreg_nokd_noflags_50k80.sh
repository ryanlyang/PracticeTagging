#!/usr/bin/env bash
#SBATCH --job-name=ab9ZAux0
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=offline_reconstructor_logs/ablations/ab9_zero_recoaux_noflags_%j.out
#SBATCH --error=offline_reconstructor_logs/ablations/ab9_zero_recoaux_noflags_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/ablations

RUN_NAME="${RUN_NAME:-ab9_zero_recoaux_jetreg_nokd_noflags_50k_80c}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_ablations}"
N_TRAIN_JETS="${N_TRAIN_JETS:-50000}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
OFFSET_JETS="${OFFSET_JETS:-0}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-20260315}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

python offline_reconstructor_joint_dualview.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --offset_jets "${OFFSET_JETS}" \
  --num_workers "${NUM_WORKERS}" \
  --enable_jet_regressor \
  --disable_final_kd \
  --loss_w_pt_ratio 0.0 \
  --loss_w_e_ratio 0.0 \
  --loss_w_budget 0.0 \
  --loss_w_sparse 0.0 \
  --loss_w_local 0.0 \
  --lambda_reco 0.25 \
  --lambda_cons 0.0 \
  --device cuda
