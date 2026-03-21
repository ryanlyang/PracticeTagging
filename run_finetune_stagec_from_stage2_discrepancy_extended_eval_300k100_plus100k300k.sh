#!/usr/bin/env bash
# Stage-C discrepancy continuation from saved Stage2 with extended eval splits:
# - keep original Stage2 train split fixed
# - extend validation by +100k jets
# - extend test by +300k jets
# - run built-in disagreement summaries on exact extended test split
#
# Submit:
#   sbatch run_finetune_stagec_from_stage2_discrepancy_extended_eval_300k100_plus100k300k.sh

#SBATCH -J stgCDiscX
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o offline_reconstructor_logs/stagec_discrepancy_extended/stagec_discext_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_discrepancy_extended/stagec_discext_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/stagec_discrepancy_extended}"
mkdir -p "${LOG_DIR}"

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_300kJ100C}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_discrepancy_extended}"
RUN_NAME="${RUN_NAME:-stagec_discrepancy_extval100k_exttest300k_seed0}"

# Keep source setup deterministic.
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

# Requested extension.
EXTRA_VAL_JETS="${EXTRA_VAL_JETS:-100000}"
EXTRA_TEST_JETS="${EXTRA_TEST_JETS:-300000}"

# Stage-C schedule.
STAGEC_EPOCHS="${STAGEC_EPOCHS:-70}"
STAGEC_FREEZE_EPOCHS="${STAGEC_FREEZE_EPOCHS:-20}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-12}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

# Discrepancy setup (smooth delta style).
DISC_TARGET_TPR="${DISC_TARGET_TPR:-0.50}"
DISC_TAU="${DISC_TAU:-0.05}"
DISC_LAMBDA="${DISC_LAMBDA:-1.0}"
DISC_MAX_MULT="${DISC_MAX_MULT:-3.0}"
DISC_TEACHER_CONF_MIN="${DISC_TEACHER_CONF_MIN:-0.60}"
DISC_CORRECTNESS_TAU="${DISC_CORRECTNESS_TAU:-0.05}"
DISC_MODE="${DISC_MODE:-smooth_delta}"

# Built-in disagreement export target.
DISAGREE_TARGET_TPR="${DISAGREE_TARGET_TPR:-0.50}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_RUN_DIR="${SAVE_DIR}/${RUN_NAME}"

echo "============================================================"
echo "Stage-C discrepancy continuation with extended val/test"
echo "Source run: ${RUN_DIR}"
echo "Output run: ${OUT_RUN_DIR}"
echo "Split extension: +val=${EXTRA_VAL_JETS}, +test=${EXTRA_TEST_JETS}"
echo "Stage-C: epochs=${STAGEC_EPOCHS}, freeze=${STAGEC_FREEZE_EPOCHS}, patience=${STAGEC_PATIENCE}, min_epochs=${STAGEC_MIN_EPOCHS}"
echo "LR/loss: lr_dual=${STAGEC_LR_DUAL}, lr_reco=${STAGEC_LR_RECO}, lambda_reco=${LAMBDA_RECO}, lambda_cons=${LAMBDA_CONS}, select=${SELECTION_METRIC}"
echo "Discrepancy: mode=${DISC_MODE}, target_tpr=${DISC_TARGET_TPR}, tau=${DISC_TAU}, lambda=${DISC_LAMBDA}, max_mult=${DISC_MAX_MULT}"
echo "============================================================"

python finetune_stagec_from_stage2_discrepancy_extended_eval.py \
  --run_dir "${RUN_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --extra_val_jets "${EXTRA_VAL_JETS}" \
  --extra_test_jets "${EXTRA_TEST_JETS}" \
  --stageC_epochs "${STAGEC_EPOCHS}" \
  --stageC_freeze_reco_epochs "${STAGEC_FREEZE_EPOCHS}" \
  --stageC_patience "${STAGEC_PATIENCE}" \
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}" \
  --stageC_lr_dual "${STAGEC_LR_DUAL}" \
  --stageC_lr_reco "${STAGEC_LR_RECO}" \
  --lambda_reco "${LAMBDA_RECO}" \
  --lambda_cons "${LAMBDA_CONS}" \
  --selection_metric "${SELECTION_METRIC}" \
  --disc_weight_enable \
  --disc_weight_mode "${DISC_MODE}" \
  --disc_target_tpr "${DISC_TARGET_TPR}" \
  --disc_tau "${DISC_TAU}" \
  --disc_lambda "${DISC_LAMBDA}" \
  --disc_max_mult "${DISC_MAX_MULT}" \
  --disc_teacher_conf_min "${DISC_TEACHER_CONF_MIN}" \
  --disc_correctness_tau "${DISC_CORRECTNESS_TAU}" \
  --disc_apply_to_reco \
  --disagreement_target_tpr "${DISAGREE_TARGET_TPR}" \
  --device "${DEVICE}"

echo
echo "Done."
echo "Metrics: ${OUT_RUN_DIR}/stagec_refine_metrics.json"
echo "Disagreement summaries: ${OUT_RUN_DIR}/disagreement_fpr50_extended_testsplit"
