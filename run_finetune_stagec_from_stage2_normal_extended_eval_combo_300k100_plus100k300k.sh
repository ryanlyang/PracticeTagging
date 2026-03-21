#!/usr/bin/env bash
# Stage-C NORMAL (no discrepancy weighting) continuation from saved Stage2,
# with extended val/test and 3-model combo search (Teacher+HLT+Joint).
#
# Submit:
#   sbatch run_finetune_stagec_from_stage2_normal_extended_eval_combo_300k100_plus100k300k.sh

#SBATCH -J stgCNormC
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o offline_reconstructor_logs/stagec_normal_extended_combo/stagec_normcombo_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_normal_extended_combo/stagec_normcombo_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/stagec_normal_extended_combo}"
mkdir -p "${LOG_DIR}"

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_300kJ100C}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_normal_extended_combo}"
RUN_NAME="${RUN_NAME:-stagec_normal_extval100k_exttest300k_combo_seed0}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

EXTRA_VAL_JETS="${EXTRA_VAL_JETS:-100000}"
EXTRA_TEST_JETS="${EXTRA_TEST_JETS:-300000}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-70}"
STAGEC_FREEZE_EPOCHS="${STAGEC_FREEZE_EPOCHS:-20}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-12}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

TARGET_TPR="${TARGET_TPR:-0.50}"
COMBO_WEIGHT_STEP="${COMBO_WEIGHT_STEP:-0.05}"
COMBO_MIN_WEIGHT="${COMBO_MIN_WEIGHT:-0.05}"
COMBO_TOP_K="${COMBO_TOP_K:-20}"

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
echo "Stage-C NORMAL continuation with extended val/test + combo search"
echo "Source run: ${RUN_DIR}"
echo "Output run: ${OUT_RUN_DIR}"
echo "Split extension: +val=${EXTRA_VAL_JETS}, +test=${EXTRA_TEST_JETS}"
echo "Stage-C: epochs=${STAGEC_EPOCHS}, freeze=${STAGEC_FREEZE_EPOCHS}, patience=${STAGEC_PATIENCE}, min_epochs=${STAGEC_MIN_EPOCHS}"
echo "LR/loss: lr_dual=${STAGEC_LR_DUAL}, lr_reco=${STAGEC_LR_RECO}, lambda_reco=${LAMBDA_RECO}, lambda_cons=${LAMBDA_CONS}, select=${SELECTION_METRIC}"
echo "Combo search: target_tpr=${TARGET_TPR}, step=${COMBO_WEIGHT_STEP}, min_weight=${COMBO_MIN_WEIGHT}, top_k=${COMBO_TOP_K}"
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
  --disagreement_target_tpr "${TARGET_TPR}" \
  --combo_search_enable \
  --combo_weight_step "${COMBO_WEIGHT_STEP}" \
  --combo_min_weight "${COMBO_MIN_WEIGHT}" \
  --combo_top_k "${COMBO_TOP_K}" \
  --device "${DEVICE}"

echo
echo "Done."
echo "Metrics: ${OUT_RUN_DIR}/stagec_refine_metrics.json"
echo "Disagreement+combo outputs: ${OUT_RUN_DIR}/disagreement_fpr50_extended_testsplit"
