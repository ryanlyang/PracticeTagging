#!/usr/bin/env bash
#SBATCH --job-name=uoSm100
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-15:00:00
#SBATCH --output=offline_reconstructor_logs/unsmear_only_smear100_300k60k500k_100c_%j.out
#SBATCH --error=offline_reconstructor_logs/unsmear_only_smear100_300k60k500k_100c_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_unsmear_only_smear100_300k60k500k_100c}"
N_TRAIN_JETS="${N_TRAIN_JETS:-860000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-300000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-60000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-500000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
SMEAR_SCALE="${SMEAR_SCALE:-1.0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running unsmear-only (smear-only HLT, smear_scale=${SMEAR_SCALE})"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unsmearonly.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06 --smear_scale ${SMEAR_SCALE} --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unsmearonly.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --n_train_split "${N_TRAIN_SPLIT}" \
  --n_val_split "${N_VAL_SPLIT}" \
  --n_test_split "${N_TEST_SPLIT}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --stageC_lr_dual 1e-5 \
  --stageC_lr_reco 5e-6 \
  --lambda_reco 0.4 \
  --lambda_cons 0.06 \
  --smear_scale "${SMEAR_SCALE}" \
  --device cuda
