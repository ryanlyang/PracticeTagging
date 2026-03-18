#!/usr/bin/env bash
# Non-privileged unmerge-only variant with Stage-B half-train curriculum:
# - Stage A (reconstructor pretrain): full train split
# - Stage B (dual pretrain, frozen reconstructor): seeded random half of train split
# - Stage C (joint finetune): full train split
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_1MJ100C_stagebhalf.sh

#SBATCH --job-name=nrivUO9Bh
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=offline_reconstructor_logs/1M_100_offline_reco_nopriv_uo_rho090_stagebhalf_%j.out
#SBATCH --error=offline_reconstructor_logs/1M_100_offline_reco_nopriv_uo_rho090_stagebhalf_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_1MJ100C_stagebhalf}"
N_TRAIN_JETS="${N_TRAIN_JETS:-1000000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"s
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-0.90}"
STAGEB_TRAIN_FRAC="${STAGEB_TRAIN_FRAC:-0.5}"
STAGEB_SUBSET_SEED="${STAGEB_SUBSET_SEED:-0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running nopriv-unmergeonly (scaled true_added target) with Stage-B half-train curriculum:"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageB_train_frac ${STAGEB_TRAIN_FRAC} --stageB_subset_seed ${STAGEB_SUBSET_SEED} --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06 --added_target_scale ${ADDED_TARGET_SCALE} --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.00 \
  --stageB_train_frac "${STAGEB_TRAIN_FRAC}" \
  --stageB_subset_seed "${STAGEB_SUBSET_SEED}" \
  --stageC_lr_dual 1e-5 \
  --stageC_lr_reco 5e-6 \
  --lambda_reco 0.4 \
  --lambda_cons 0.06 \
  --added_target_scale "${ADDED_TARGET_SCALE}" \
  --disable_final_kd \
  --device cuda
