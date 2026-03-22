#!/usr/bin/env bash
#SBATCH --job-name=uoTokResCal
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/unsmear_only_tokenres_calib_300k60k500k_100c_%j.out
#SBATCH --error=offline_reconstructor_logs/unsmear_only_tokenres_calib_300k60k500k_100c_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_unsmear_only_tokenres_calib_300k60k500k_100c}"
N_TRAIN_JETS="${N_TRAIN_JETS:-30000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-18000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-6000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-6000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

# Calibrated smearing knobs
SMEAR_SCALE="${SMEAR_SCALE:-1.5}"
SMEAR_CORE_SCALE="${SMEAR_CORE_SCALE:-1.5}"
SMEAR_ANGLE_SCALE="${SMEAR_ANGLE_SCALE:-1.5}"
SMEAR_TAIL_BASE="${SMEAR_TAIL_BASE:-0.04}"
SMEAR_TAIL_SIGMA_MULT="${SMEAR_TAIL_SIGMA_MULT:-2.0}"
SMEAR_TAIL_PROB_MAX="${SMEAR_TAIL_PROB_MAX:-0.35}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running unsmear-only token-residual (calibrated smearing, GeV-mode)"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unsmearonly.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06 --smear_scale ${SMEAR_SCALE} --smear_use_pt_gev --smear_core_scale ${SMEAR_CORE_SCALE} --smear_angle_scale ${SMEAR_ANGLE_SCALE} --smear_tail_base ${SMEAR_TAIL_BASE} --smear_tail_sigma_mult ${SMEAR_TAIL_SIGMA_MULT} --smear_tail_prob_max ${SMEAR_TAIL_PROB_MAX} --unsmear_loss_mode token_residual --token_match_dr_max 0.08 --token_match_dlogpt_max 1.40 --token_match_logpt_alpha 0.20 --token_res_w_logpt 1.0 --token_res_w_eta 0.35 --token_res_w_phi 0.35 --token_res_w_logE 0.70 --token_cov_penalty 0.05 --token_jet_w_pt_ratio 0.12 --token_jet_w_e_ratio 0.12 --device cuda"

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
  --smear_use_pt_gev \
  --smear_core_scale "${SMEAR_CORE_SCALE}" \
  --smear_angle_scale "${SMEAR_ANGLE_SCALE}" \
  --smear_tail_base "${SMEAR_TAIL_BASE}" \
  --smear_tail_sigma_mult "${SMEAR_TAIL_SIGMA_MULT}" \
  --smear_tail_prob_max "${SMEAR_TAIL_PROB_MAX}" \
  --unsmear_loss_mode token_residual \
  --token_match_dr_max 0.08 \
  --token_match_dlogpt_max 1.40 \
  --token_match_logpt_alpha 0.20 \
  --token_res_w_logpt 1.0 \
  --token_res_w_eta 0.35 \
  --token_res_w_phi 0.35 \
  --token_res_w_logE 0.70 \
  --token_cov_penalty 0.05 \
  --token_jet_w_pt_ratio 0.12 \
  --token_jet_w_e_ratio 0.12 \
  --device cuda
