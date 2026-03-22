#!/usr/bin/env bash
#SBATCH --job-name=uoTAKDcal
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/unsmear_only_stageAteacherkd_calib_300k60k500k_100c_%j.out
#SBATCH --error=offline_reconstructor_logs/unsmear_only_stageAteacherkd_calib_300k60k500k_100c_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_unsmear_only_stageAteacherkd_calib_300k60k500k_100c}"
N_TRAIN_JETS="${N_TRAIN_JETS:-860000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-300000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-60000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-500000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

# Calibrated smearing knobs (normal baseline scale, not 1.25x)
SMEAR_SCALE="${SMEAR_SCALE:-1.5}"
SMEAR_CORE_SCALE="${SMEAR_CORE_SCALE:-1.5}"
SMEAR_ANGLE_SCALE="${SMEAR_ANGLE_SCALE:-1.5}"
SMEAR_TAIL_BASE="${SMEAR_TAIL_BASE:-0.04}"
SMEAR_TAIL_SIGMA_MULT="${SMEAR_TAIL_SIGMA_MULT:-2.0}"
SMEAR_TAIL_PROB_MAX="${SMEAR_TAIL_PROB_MAX:-0.35}"

# Stage-A teacher-KD / budget-hinge knobs.
STAGEA_KD_TEMP="${STAGEA_KD_TEMP:-2.5}"
STAGEA_LAMBDA_KD="${STAGEA_LAMBDA_KD:-5.0}"
STAGEA_LAMBDA_PHYS="${STAGEA_LAMBDA_PHYS:-0.05}"
STAGEA_LAMBDA_BUDGET_HINGE="${STAGEA_LAMBDA_BUDGET_HINGE:-1.0}"
STAGEA_BUDGET_EPS="${STAGEA_BUDGET_EPS:-0.015}"
STAGEA_BUDGET_WEIGHT_FLOOR="${STAGEA_BUDGET_WEIGHT_FLOOR:-1e-4}"
STAGEA_TARGET_TPR="${STAGEA_TARGET_TPR:-0.50}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running unsmear-only Stage-A teacher-KD (calibrated smearing, GeV-mode)"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unsmearonly_stageAteacherkd.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06 --smear_scale ${SMEAR_SCALE} --smear_use_pt_gev --smear_core_scale ${SMEAR_CORE_SCALE} --smear_angle_scale ${SMEAR_ANGLE_SCALE} --smear_tail_base ${SMEAR_TAIL_BASE} --smear_tail_sigma_mult ${SMEAR_TAIL_SIGMA_MULT} --smear_tail_prob_max ${SMEAR_TAIL_PROB_MAX} --stageA_kd_temp ${STAGEA_KD_TEMP} --stageA_lambda_kd ${STAGEA_LAMBDA_KD} --stageA_lambda_phys ${STAGEA_LAMBDA_PHYS} --stageA_lambda_budget_hinge ${STAGEA_LAMBDA_BUDGET_HINGE} --stageA_budget_eps ${STAGEA_BUDGET_EPS} --stageA_budget_weight_floor ${STAGEA_BUDGET_WEIGHT_FLOOR} --stageA_target_tpr ${STAGEA_TARGET_TPR} --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unsmearonly_stageAteacherkd.py \
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
  --smear_use_pt_gev \
  --smear_core_scale "${SMEAR_CORE_SCALE}" \
  --smear_angle_scale "${SMEAR_ANGLE_SCALE}" \
  --smear_tail_base "${SMEAR_TAIL_BASE}" \
  --smear_tail_sigma_mult "${SMEAR_TAIL_SIGMA_MULT}" \
  --smear_tail_prob_max "${SMEAR_TAIL_PROB_MAX}" \
  --stageA_kd_temp "${STAGEA_KD_TEMP}" \
  --stageA_lambda_kd "${STAGEA_LAMBDA_KD}" \
  --stageA_lambda_phys "${STAGEA_LAMBDA_PHYS}" \
  --stageA_lambda_budget_hinge "${STAGEA_LAMBDA_BUDGET_HINGE}" \
  --stageA_budget_eps "${STAGEA_BUDGET_EPS}" \
  --stageA_budget_weight_floor "${STAGEA_BUDGET_WEIGHT_FLOOR}" \
  --stageA_target_tpr "${STAGEA_TARGET_TPR}" \
  --device cuda
