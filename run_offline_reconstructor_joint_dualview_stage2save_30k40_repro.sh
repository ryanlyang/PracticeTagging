#!/usr/bin/env bash
# 30k/40c repro runner for offline_reconstructor_joint_dualview_stage2save.py
# Saves Stage-B checkpoints/metrics in addition to final joint results.
#SBATCH --job-name=offrecoS2
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_stage2save_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_stage2save_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_30k_40c_stage2save}"
N_TRAIN_JETS="${N_TRAIN_JETS:-30000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-40}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
ENABLE_JET_REGRESSOR="${ENABLE_JET_REGRESSOR:-0}"

JET_REG_ARGS=()
if [ "${ENABLE_JET_REGRESSOR}" = "1" ]; then
  JET_REG_ARGS+=(--enable_jet_regressor)
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running stage2-save repro config:"
echo "python offline_reconstructor_joint_dualview_stage2save.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --disable_final_kd --device cuda ${JET_REG_ARGS[*]}"

python offline_reconstructor_joint_dualview_stage2save.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --disable_final_kd \
  --device cuda \
  "${JET_REG_ARGS[@]}"
