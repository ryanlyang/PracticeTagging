#!/usr/bin/env bash
set -euo pipefail

# Shared runner for Stage-A loss sweeps focused on teacher-on-reco performance.

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stageA_teacher_auc_sweep10}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
OFFSET_JETS="${OFFSET_JETS:-0}"

# User-requested split sizes.
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-500000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-100000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-500000}"
N_TRAIN_JETS="${N_TRAIN_JETS:-1100000}"

ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-0.90}"

# Keep Stage-B/C enabled for compatibility with existing pipeline unless overridden.
STAGEB_EPOCHS="${STAGEB_EPOCHS:-45}"
STAGEB_PATIENCE="${STAGEB_PATIENCE:-12}"
STAGEB_MIN_EPOCHS="${STAGEB_MIN_EPOCHS:-12}"
STAGEB_LR_DUAL="${STAGEB_LR_DUAL:-4e-4}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-65}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"

# Stage-A sweep knobs.
STAGEA_EPOCHS="${STAGEA_EPOCHS:-90}"
STAGEA_PATIENCE="${STAGEA_PATIENCE:-18}"
STAGEA_KD_TEMP="${STAGEA_KD_TEMP:-2.5}"
STAGEA_LAMBDA_KD="${STAGEA_LAMBDA_KD:-1.0}"
STAGEA_LAMBDA_EMB="${STAGEA_LAMBDA_EMB:-1.2}"
STAGEA_LAMBDA_TOK="${STAGEA_LAMBDA_TOK:-0.6}"
STAGEA_LAMBDA_PHYS="${STAGEA_LAMBDA_PHYS:-0.2}"
STAGEA_LAMBDA_BUDGET_HINGE="${STAGEA_LAMBDA_BUDGET_HINGE:-0.03}"
STAGEA_BUDGET_EPS="${STAGEA_BUDGET_EPS:-0.015}"
STAGEA_BUDGET_WEIGHT_FLOOR="${STAGEA_BUDGET_WEIGHT_FLOOR:-1e-4}"
STAGEA_TARGET_TPR="${STAGEA_TARGET_TPR:-0.50}"
STAGEA_LOSS_NORM_EMA_DECAY="${STAGEA_LOSS_NORM_EMA_DECAY:-0.98}"
STAGEA_LOSS_NORM_EPS="${STAGEA_LOSS_NORM_EPS:-1e-6}"
DISABLE_STAGEA_LOSS_NORMALIZATION="${DISABLE_STAGEA_LOSS_NORMALIZATION:-0}"

REPORT_TARGET_TPR="${REPORT_TARGET_TPR:-0.50}"
COMBO_WEIGHT_STEP="${COMBO_WEIGHT_STEP:-0.01}"
DIAG_MATCH_MAX_JETS="${DIAG_MATCH_MAX_JETS:-20000}"
DIAG_MATCH_SEED="${DIAG_MATCH_SEED:--1}"

ANALYZE_DEVICE="${ANALYZE_DEVICE:-cuda}"
ANALYZE_EVAL_BATCH_SIZE="${ANALYZE_EVAL_BATCH_SIZE:-512}"
ANALYZE_RECO_BATCH_SIZE="${ANALYZE_RECO_BATCH_SIZE:-256}"
ANALYZE_NUM_WORKERS="${ANALYZE_NUM_WORKERS:-1}"
ANALYZE_OUTPUT_NAME="${ANALYZE_OUTPUT_NAME:-hlt_vs_teacher_on_stage2_reco_overlap_tpr50.json}"
DATA_FILE="${DATA_FILE:-}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "${SAVE_DIR}"

EXTRA_NORM_FLAG=()
if [[ "${DISABLE_STAGEA_LOSS_NORMALIZATION}" == "1" ]]; then
  EXTRA_NORM_FLAG+=(--disable_stageA_loss_normalization)
fi

TRAIN_CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --selection_metric auc
  --stageA_epochs "${STAGEA_EPOCHS}"
  --stageA_patience "${STAGEA_PATIENCE}"
  --stageA_kd_temp "${STAGEA_KD_TEMP}"
  --stageA_lambda_kd "${STAGEA_LAMBDA_KD}"
  --stageA_lambda_emb "${STAGEA_LAMBDA_EMB}"
  --stageA_lambda_tok "${STAGEA_LAMBDA_TOK}"
  --stageA_lambda_phys "${STAGEA_LAMBDA_PHYS}"
  --stageA_lambda_budget_hinge "${STAGEA_LAMBDA_BUDGET_HINGE}"
  --stageA_budget_eps "${STAGEA_BUDGET_EPS}"
  --stageA_budget_weight_floor "${STAGEA_BUDGET_WEIGHT_FLOOR}"
  --stageA_target_tpr "${STAGEA_TARGET_TPR}"
  --stageA_loss_norm_ema_decay "${STAGEA_LOSS_NORM_EMA_DECAY}"
  --stageA_loss_norm_eps "${STAGEA_LOSS_NORM_EPS}"
  --stageB_epochs "${STAGEB_EPOCHS}"
  --stageB_patience "${STAGEB_PATIENCE}"
  --stageB_min_epochs "${STAGEB_MIN_EPOCHS}"
  --stageB_lr_dual "${STAGEB_LR_DUAL}"
  --stageB_lambda_rank 0.0
  --stageB_lambda_cons 0.0
  --stageC_epochs "${STAGEC_EPOCHS}"
  --stageC_patience "${STAGEC_PATIENCE}"
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}"
  --stageC_lr_dual "${STAGEC_LR_DUAL}"
  --stageC_lr_reco "${STAGEC_LR_RECO}"
  --lambda_reco "${LAMBDA_RECO}"
  --lambda_cons "${LAMBDA_CONS}"
  --added_target_scale "${ADDED_TARGET_SCALE}"
  --report_target_tpr "${REPORT_TARGET_TPR}"
  --combo_weight_step "${COMBO_WEIGHT_STEP}"
  --diag_match_max_jets "${DIAG_MATCH_MAX_JETS}"
  --diag_match_seed "${DIAG_MATCH_SEED}"
  --disable_final_kd
  --device "${DEVICE}"
)

if [[ ${#EXTRA_NORM_FLAG[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_NORM_FLAG[@]}")
fi

echo "============================================================"
echo "Stage-A Sweep Run"
echo "Run name: ${RUN_NAME}"
echo "Save dir: ${SAVE_DIR}/${RUN_NAME}"
echo "Splits  : train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "Loss    : kd=${STAGEA_LAMBDA_KD}, emb=${STAGEA_LAMBDA_EMB}, tok=${STAGEA_LAMBDA_TOK}, phys=${STAGEA_LAMBDA_PHYS}, budget_hinge=${STAGEA_LAMBDA_BUDGET_HINGE}, T=${STAGEA_KD_TEMP}, norm_off=${DISABLE_STAGEA_LOSS_NORMALIZATION}"
echo "============================================================"
printf ' %q' "${TRAIN_CMD[@]}"
echo
"${TRAIN_CMD[@]}"

RUN_DIR_FULL="${SAVE_DIR}/${RUN_NAME}"
ANALYZE_CMD=(
  python analyze_hlt_vs_teacher_stage2_reco_overlap.py
  --run_dir "${RUN_DIR_FULL}"
  --target_tpr "${REPORT_TARGET_TPR}"
  --combo_weight_step "${COMBO_WEIGHT_STEP}"
  --device "${ANALYZE_DEVICE}"
  --eval_batch_size "${ANALYZE_EVAL_BATCH_SIZE}"
  --reco_batch_size "${ANALYZE_RECO_BATCH_SIZE}"
  --num_workers "${ANALYZE_NUM_WORKERS}"
  --output_name "${ANALYZE_OUTPUT_NAME}"
)
if [[ -n "${DATA_FILE}" ]]; then
  ANALYZE_CMD+=(--data_file "${DATA_FILE}")
fi

echo "Post-run overlap/combo analysis..."
printf ' %q' "${ANALYZE_CMD[@]}"
echo
"${ANALYZE_CMD[@]}"

echo "Done: ${RUN_DIR_FULL}"
