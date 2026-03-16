#!/usr/bin/env bash
# 100k/80 run with Stage-B "earliest epoch within AUC tolerance of best" handoff.
# Uses: offline_reconstructor_joint_dualview_stage2save_auc_norankc_stagebearly.py
#SBATCH --job-name=offrecoAEB
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_auceb_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_auceb_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stagebearly_auc}"
N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
ENABLE_JET_REGRESSOR="${ENABLE_JET_REGRESSOR:-0}"
STAGEB_AUC_EARLY_TOL="${STAGEB_AUC_EARLY_TOL:-0.001}"

JET_REG_ARGS=()
if [ "${ENABLE_JET_REGRESSOR}" = "1" ]; then
  JET_REG_ARGS+=(--enable_jet_regressor)
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "Running Stage-B early-AUC handoff config:"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_stagebearly.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_auc_early_tol ${STAGEB_AUC_EARLY_TOL} --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --disable_final_kd --device cuda ${JET_REG_ARGS[*]}"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_stagebearly.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric auc \
  --stageB_auc_early_tol "${STAGEB_AUC_EARLY_TOL}" \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --disable_final_kd \
  --device cuda \
  "${JET_REG_ARGS[@]}"
