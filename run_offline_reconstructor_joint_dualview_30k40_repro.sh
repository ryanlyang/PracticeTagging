#!/usr/bin/env bash
# Reproduce the old "joint_50k_80c" style run:
# - 30k jets
# - 40 max constituents
# - no jet regressor
# - no final KD stage
# (The old folder name was misleading; hlt_stats shows n_jets=30000.)
#SBATCH --job-name=offrecoJ3040
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_30k40_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_30k40_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_30k_40c_repro}"
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

echo "Running repro config:"
echo "python offline_reconstructor_joint_dualview.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --disable_final_kd \
  --device cuda

