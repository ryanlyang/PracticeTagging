#!/usr/bin/env bash
#SBATCH --job-name=ab5StCKD
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=offline_reconstructor_logs/ablations/ab5_stagec_kd_%j.out
#SBATCH --error=offline_reconstructor_logs/ablations/ab5_stagec_kd_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/ablations

RUN_NAME="${RUN_NAME:-ab5_stagec_kd_50k_80c}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_ablations}"
N_TRAIN_JETS="${N_TRAIN_JETS:-50000}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
OFFSET_JETS="${OFFSET_JETS:-0}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-20260315}"
LAMBDA_KD_STAGEC="${LAMBDA_KD_STAGEC:-0.20}"
KD_TEMP="${KD_TEMP:-7.0}"

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
  --stageC_enable_kd \
  --lambda_kd_stageC "${LAMBDA_KD_STAGEC}" \
  --stageC_kd_temperature "${KD_TEMP}" \
  --stageC_kd_conf_weighted \
  --enable_jet_regressor \
  --disable_final_kd \
  --device cuda
