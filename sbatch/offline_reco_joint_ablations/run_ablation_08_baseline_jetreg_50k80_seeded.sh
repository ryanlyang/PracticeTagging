#!/usr/bin/env bash
#SBATCH --job-name=ab8BaseJR
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=offline_reconstructor_logs/ablations/ab8_base_jetreg_%j.out
#SBATCH --error=offline_reconstructor_logs/ablations/ab8_base_jetreg_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/ablations

RUN_NAME="${RUN_NAME:-ab8_baseline_jetreg_50k_80c_seeded}"
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

python offline_reconstructor_joint_dualview_ablations.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --offset_jets "${OFFSET_JETS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --enable_jet_regressor \
  --device cuda
