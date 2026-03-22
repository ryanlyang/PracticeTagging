#!/usr/bin/env bash
# Stage-C specialist continuation anchored to NORMAL joint, with:
# - push-hard discrepancy weighting (teacher vs anchor-normal residuals)
# - checkpoint selection by best normal+special weighted combo FPR@target TPR on val
# - post-run two-model weighted sweep + OR-union TPR-pair sweep on test
#
# Submit:
#   sbatch run_finetune_stagec_specialist_against_normal_extended_eval_combo_union_300k100.sh

#SBATCH -J stgCSpecN
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 15:00:00
#SBATCH -o offline_reconstructor_logs/stagec_specialist_against_normal/stagec_specnorm_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_specialist_against_normal/stagec_specnorm_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/stagec_specialist_against_normal}"
mkdir -p "${LOG_DIR}"

# Use the previously trained NORMAL continuation run as source + anchor.
RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_normal_extended_combo/stagec_normal_extval100k_exttest300k_combo_seed0}"
ANCHOR_RUN_DIR="${ANCHOR_RUN_DIR:-${RUN_DIR}}"

SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_specialist_against_normal}"
RUN_NAME="${RUN_NAME:-stagec_specialist_pushhard_anchorNormal_comboSel_union_seed0}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

# Keep same loaded data domain by default (source run already has extended split).
EXTRA_VAL_JETS="${EXTRA_VAL_JETS:-0}"
EXTRA_TEST_JETS="${EXTRA_TEST_JETS:-0}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-70}"
STAGEC_FREEZE_EPOCHS="${STAGEC_FREEZE_EPOCHS:-20}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-12}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"

# Deploy-oriented checkpoint selection: best normal+special combo FPR@target TPR on val.
SELECTION_METRIC="${SELECTION_METRIC:-combo_fpr50}"
VAL_COMBO_TARGET_TPR="${VAL_COMBO_TARGET_TPR:-0.50}"
VAL_COMBO_WEIGHT_STEP="${VAL_COMBO_WEIGHT_STEP:-0.01}"
VAL_COMBO_MIN_WEIGHT="${VAL_COMBO_MIN_WEIGHT:-0.00}"

# Push-hard discrepancy weighting against anchor-normal.
DISC_TARGET_TPR="${DISC_TARGET_TPR:-0.50}"
DISC_MODE="${DISC_MODE:-tail_disagreement}"
DISC_TAU="${DISC_TAU:-0.025}"
DISC_LAMBDA="${DISC_LAMBDA:-6.0}"
DISC_MAX_MULT="${DISC_MAX_MULT:-8.0}"
DISC_INCLUDE_POS="${DISC_INCLUDE_POS:-1}"
DISC_POS_SCALE="${DISC_POS_SCALE:-0.50}"
DISC_TEACHER_CONF_MIN="${DISC_TEACHER_CONF_MIN:-0.60}"
DISC_CORRECTNESS_TAU="${DISC_CORRECTNESS_TAU:-0.05}"

TARGET_TPR="${TARGET_TPR:-0.50}"

# Two-model weighted search (normal+special).
TWO_MODEL_COMBO_ENABLE="${TWO_MODEL_COMBO_ENABLE:-1}"
TWO_MODEL_COMBO_WEIGHT_STEP="${TWO_MODEL_COMBO_WEIGHT_STEP:-0.01}"
TWO_MODEL_COMBO_MIN_WEIGHT="${TWO_MODEL_COMBO_MIN_WEIGHT:-0.00}"
TWO_MODEL_COMBO_TOP_K="${TWO_MODEL_COMBO_TOP_K:-80}"

# Two-model OR-union TPR-pair search.
TWO_MODEL_UNION_ENABLE="${TWO_MODEL_UNION_ENABLE:-1}"
TWO_MODEL_UNION_TPR_MIN="${TWO_MODEL_UNION_TPR_MIN:-0.35}"
TWO_MODEL_UNION_TPR_MAX="${TWO_MODEL_UNION_TPR_MAX:-0.60}"
TWO_MODEL_UNION_TPR_STEP="${TWO_MODEL_UNION_TPR_STEP:-0.01}"
TWO_MODEL_UNION_TPR_TOL="${TWO_MODEL_UNION_TPR_TOL:-0.003}"
TWO_MODEL_UNION_TOP_K="${TWO_MODEL_UNION_TOP_K:-120}"

# Keep original 3-model search off by default; enable if you want teacher/HLT/special oracle check.
COMBO_SEARCH_ENABLE="${COMBO_SEARCH_ENABLE:-0}"
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
echo "Stage-C SPECIALIST anchored to NORMAL"
echo "Source run: ${RUN_DIR}"
echo "Anchor run: ${ANCHOR_RUN_DIR}"
echo "Output run: ${OUT_RUN_DIR}"
echo "Split extension: +val=${EXTRA_VAL_JETS}, +test=${EXTRA_TEST_JETS}"
echo "Stage-C: epochs=${STAGEC_EPOCHS}, freeze=${STAGEC_FREEZE_EPOCHS}, patience=${STAGEC_PATIENCE}, min_epochs=${STAGEC_MIN_EPOCHS}"
echo "Selection: ${SELECTION_METRIC} (target_tpr=${VAL_COMBO_TARGET_TPR}, step=${VAL_COMBO_WEIGHT_STEP})"
echo "Discrepancy push-hard: mode=${DISC_MODE}, tau=${DISC_TAU}, lambda=${DISC_LAMBDA}, max_mult=${DISC_MAX_MULT}, include_pos=${DISC_INCLUDE_POS}, pos_scale=${DISC_POS_SCALE}"
echo "============================================================"

CMD=(
  python finetune_stagec_from_stage2_specialist_against_normal_extended_eval.py
  --run_dir "${RUN_DIR}"
  --anchor_run_dir "${ANCHOR_RUN_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --max_constits "${MAX_CONSTITS}"
  --num_workers "${NUM_WORKERS}"
  --extra_val_jets "${EXTRA_VAL_JETS}"
  --extra_test_jets "${EXTRA_TEST_JETS}"
  --stageC_epochs "${STAGEC_EPOCHS}"
  --stageC_freeze_reco_epochs "${STAGEC_FREEZE_EPOCHS}"
  --stageC_patience "${STAGEC_PATIENCE}"
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}"
  --stageC_lr_dual "${STAGEC_LR_DUAL}"
  --stageC_lr_reco "${STAGEC_LR_RECO}"
  --lambda_reco "${LAMBDA_RECO}"
  --lambda_cons "${LAMBDA_CONS}"
  --selection_metric "${SELECTION_METRIC}"
  --val_combo_target_tpr "${VAL_COMBO_TARGET_TPR}"
  --val_combo_weight_step "${VAL_COMBO_WEIGHT_STEP}"
  --val_combo_min_weight "${VAL_COMBO_MIN_WEIGHT}"
  --disc_weight_enable
  --disc_weight_mode "${DISC_MODE}"
  --disc_target_tpr "${DISC_TARGET_TPR}"
  --disc_tau "${DISC_TAU}"
  --disc_lambda "${DISC_LAMBDA}"
  --disc_max_mult "${DISC_MAX_MULT}"
  --disc_pos_scale "${DISC_POS_SCALE}"
  --disc_teacher_conf_min "${DISC_TEACHER_CONF_MIN}"
  --disc_correctness_tau "${DISC_CORRECTNESS_TAU}"
  --disc_apply_to_reco
  --disagreement_target_tpr "${TARGET_TPR}"
  --two_model_combo_weight_step "${TWO_MODEL_COMBO_WEIGHT_STEP}"
  --two_model_combo_min_weight "${TWO_MODEL_COMBO_MIN_WEIGHT}"
  --two_model_combo_top_k "${TWO_MODEL_COMBO_TOP_K}"
  --two_model_union_tpr_min "${TWO_MODEL_UNION_TPR_MIN}"
  --two_model_union_tpr_max "${TWO_MODEL_UNION_TPR_MAX}"
  --two_model_union_tpr_step "${TWO_MODEL_UNION_TPR_STEP}"
  --two_model_union_tpr_tol "${TWO_MODEL_UNION_TPR_TOL}"
  --two_model_union_top_k "${TWO_MODEL_UNION_TOP_K}"
  --device "${DEVICE}"
)

if [[ "${DISC_INCLUDE_POS}" == "1" ]]; then
  CMD+=(--disc_include_pos)
fi
if [[ "${TWO_MODEL_COMBO_ENABLE}" == "1" ]]; then
  CMD+=(--two_model_combo_enable)
fi
if [[ "${TWO_MODEL_UNION_ENABLE}" == "1" ]]; then
  CMD+=(--two_model_union_enable)
fi
if [[ "${COMBO_SEARCH_ENABLE}" == "1" ]]; then
  CMD+=(
    --combo_search_enable
    --combo_weight_step "${COMBO_WEIGHT_STEP}"
    --combo_min_weight "${COMBO_MIN_WEIGHT}"
    --combo_top_k "${COMBO_TOP_K}"
  )
fi

"${CMD[@]}"

echo
echo "Done."
echo "Metrics: ${OUT_RUN_DIR}/stagec_refine_metrics.json"
echo "Disagreement+combo outputs: ${OUT_RUN_DIR}/disagreement_fpr50_extended_testsplit"
