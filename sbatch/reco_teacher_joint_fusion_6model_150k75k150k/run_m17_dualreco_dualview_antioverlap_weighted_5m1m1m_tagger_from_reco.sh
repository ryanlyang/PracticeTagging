#!/usr/bin/env bash
#SBATCH --job-name=m17aot
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=320G
#SBATCH --time=6-00:00:00
#SBATCH --requeue
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m17_dualreco_antioverlap_tagger_from_reco_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m17_dualreco_antioverlap_tagger_from_reco_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_from_recoonly}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_from_recoonly}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"

N_TRAIN_JETS="${N_TRAIN_JETS:-7000000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-5000000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-1000000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-1000000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
RECO_PRETRAIN_DIR="${RECO_PRETRAIN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_recoonly/model17_dualreco_dualview_antioverlap_weighted_5m1m1m_seed0_recoonly}"
STEP1_LOAD_DIR="${STEP1_LOAD_DIR:-${RECO_PRETRAIN_DIR}}"
LOAD_RECO_A_CKPT="${LOAD_RECO_A_CKPT:-${RECO_PRETRAIN_DIR}/offline_reconstructor_A_stageA.pt}"
LOAD_RECO_B_CKPT="${LOAD_RECO_B_CKPT:-${RECO_PRETRAIN_DIR}/offline_reconstructor_B_stageA.pt}"
RECO_A_PRETRAIN_DIR="${RECO_A_PRETRAIN_DIR:-${RECO_PRETRAIN_DIR//_recoonly/_recoAonly}}"
RECO_B_PRETRAIN_DIR="${RECO_B_PRETRAIN_DIR:-${RECO_PRETRAIN_DIR//_recoonly/_recoBonly}}"
STEP1_FALLBACK_FROM_RECOONLY="${STEP1_FALLBACK_FROM_RECOONLY:-${RECO_PRETRAIN_DIR//_recoonly/}}"
STEP1_FALLBACK_FROM_RECOAONLY="${STEP1_FALLBACK_FROM_RECOAONLY:-${RECO_A_PRETRAIN_DIR//_recoAonly/}}"
STEP1_FALLBACK_FROM_RECOBONLY="${STEP1_FALLBACK_FROM_RECOBONLY:-${RECO_B_PRETRAIN_DIR//_recoBonly/}}"

OFFDROP_PROB_MAX="${OFFDROP_PROB_MAX:-0.0}"
RATIO_COUNT_UNDER_LAMBDA="${RATIO_COUNT_UNDER_LAMBDA:-1.0}"
RATIO_COUNT_OVER_LAMBDA="${RATIO_COUNT_OVER_LAMBDA:-0.25}"
RATIO_COUNT_MARGIN_BASE="${RATIO_COUNT_MARGIN_BASE:-2.0}"
RATIO_COUNT_MARGIN_SCALE="${RATIO_COUNT_MARGIN_SCALE:-6.0}"
RATIO_COUNT_GAMMA="${RATIO_COUNT_GAMMA:-0.70}"
RATIO_COUNT_OVER_FLOOR="${RATIO_COUNT_OVER_FLOOR:-0.05}"
RATIO_COUNT_EPS="${RATIO_COUNT_EPS:-0.015}"

TEACHER_ANTI_LAMBDA="${TEACHER_ANTI_LAMBDA:-0.02}"
TEACHER_ANTI_TAU="${TEACHER_ANTI_TAU:-0.05}"
TEACHER_ANTI_BETA="${TEACHER_ANTI_BETA:-0.10}"
TEACHER_ANTI_WARMUP_EPOCHS="${TEACHER_ANTI_WARMUP_EPOCHS:-12}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${SAVE_DIR}"

resolve_existing_file() {
  for p in "$@"; do
    if [[ -n "${p}" && -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

resolve_step1_dir() {
  for d in "$@"; do
    if [[ -n "${d}" && -f "${d}/teacher.pt" && -f "${d}/baseline.pt" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

# Auto-resolve STEP1/checkpoint paths for split recoAonly/recoBonly layout.
if _d="$(resolve_step1_dir \
  "${STEP1_LOAD_DIR}" \
  "${STEP1_FALLBACK_FROM_RECOONLY}" \
  "${STEP1_FALLBACK_FROM_RECOAONLY}" \
  "${STEP1_FALLBACK_FROM_RECOBONLY}" \
  "${RECO_PRETRAIN_DIR}" \
  "${RECO_A_PRETRAIN_DIR}" \
  "${RECO_B_PRETRAIN_DIR}")"; then
  STEP1_LOAD_DIR="${_d}"
fi
if _p="$(resolve_existing_file \
  "${LOAD_RECO_A_CKPT}" \
  "${RECO_PRETRAIN_DIR}/offline_reconstructor_A_stageA.pt" \
  "${RECO_A_PRETRAIN_DIR}/offline_reconstructor_A_stageA.pt")"; then
  LOAD_RECO_A_CKPT="${_p}"
fi
if _p="$(resolve_existing_file \
  "${LOAD_RECO_B_CKPT}" \
  "${RECO_PRETRAIN_DIR}/offline_reconstructor_B_stageA.pt" \
  "${RECO_B_PRETRAIN_DIR}/offline_reconstructor_B_stageA.pt")"; then
  LOAD_RECO_B_CKPT="${_p}"
fi
unset _d _p

echo "Resolved STEP1_LOAD_DIR=${STEP1_LOAD_DIR}"
echo "Resolved LOAD_RECO_A_CKPT=${LOAD_RECO_A_CKPT}"
echo "Resolved LOAD_RECO_B_CKPT=${LOAD_RECO_B_CKPT}"

if [[ ! -f "${STEP1_LOAD_DIR}/teacher.pt" ]]; then
  echo "ERROR: Missing teacher checkpoint: ${STEP1_LOAD_DIR}/teacher.pt" >&2
  exit 2
fi
if [[ ! -f "${STEP1_LOAD_DIR}/baseline.pt" ]]; then
  echo "ERROR: Missing baseline checkpoint: ${STEP1_LOAD_DIR}/baseline.pt" >&2
  exit 2
fi
if [[ ! -f "${LOAD_RECO_A_CKPT}" ]]; then
  echo "ERROR: Missing reco-A checkpoint: ${LOAD_RECO_A_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${LOAD_RECO_B_CKPT}" ]]; then
  echo "ERROR: Missing reco-B checkpoint: ${LOAD_RECO_B_CKPT}" >&2
  exit 2
fi

CMD=(
  python train_m9_dualreco_dualview_offdrop.py
  --train_path "${TRAIN_PATH}"
  --use_train_weights
  --force_m5_step1
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

  --target_mode offdrop

  --teacher_use_anti_overlap
  --teacher_anti_lambda "${TEACHER_ANTI_LAMBDA}"
  --teacher_anti_tau "${TEACHER_ANTI_TAU}"
  --teacher_anti_beta "${TEACHER_ANTI_BETA}"
  --teacher_anti_warmup_epochs "${TEACHER_ANTI_WARMUP_EPOCHS}"

  --stageA_epochs 90
  --stageA_patience 18
  --stageA_kd_temp 2.5
  --stageA_lambda_kd 5.0
  --stageA_lambda_emb 0.0
  --stageA_lambda_tok 0.0
  --stageA_lambda_phys 0.05
  --stageA_lambda_budget_hinge 1.0
  --stageA_budget_eps 0.015
  --stageA_budget_weight_floor 1e-4
  --stageA_target_tpr 0.50
  --stageA_lambda_delta 0.15
  --stageA_delta_tau 0.05
  --stageA_delta_lambda_fp 3.0
  --stageA_loss_norm_ema_decay 0.98
  --stageA_loss_norm_eps 1e-6
  --added_target_scale 0.90

  --target_drop_prob_max "${OFFDROP_PROB_MAX}"
  --target_drop_num_banks 1
  --target_drop_bank_cycle_epochs 1
  --recoB_epochs 90
  --recoB_patience 18
  --recoB_lr 3e-4
  --recoB_weight_decay 1e-4
  --recoB_warmup_epochs 5
  --recoB_stage1_epochs 20
  --recoB_stage2_epochs 55
  --recoB_min_full_scale_epochs 5
  --recoB_ratio_count_under_lambda "${RATIO_COUNT_UNDER_LAMBDA}"
  --recoB_ratio_count_over_lambda "${RATIO_COUNT_OVER_LAMBDA}"
  --recoB_ratio_count_over_margin_base "${RATIO_COUNT_MARGIN_BASE}"
  --recoB_ratio_count_over_margin_scale "${RATIO_COUNT_MARGIN_SCALE}"
  --recoB_ratio_count_over_ratio_gamma "${RATIO_COUNT_GAMMA}"
  --recoB_ratio_count_over_lambda_floor "${RATIO_COUNT_OVER_FLOOR}"
  --recoB_ratio_count_eps "${RATIO_COUNT_EPS}"

  --disable_recoB_ratio_budget

  --corrected_weight_floor 0.03
  --reco_eval_batch_size 256
  --select_metric auc

  --dual_frozen_epochs 45
  --dual_frozen_patience 12
  --dual_frozen_batch_size 256
  --dual_frozen_lr 3e-4
  --dual_frozen_weight_decay 1e-4
  --dual_frozen_warmup_epochs 5
  --dual_frozen_lambda_rank 0.2
  --dual_frozen_rank_tau 0.05

  --dual_joint_epochs 12
  --dual_joint_patience 6
  --dual_joint_batch_size 128
  --dual_joint_lr_dual 1e-4
  --dual_joint_lr_reco_a 2e-6
  --dual_joint_lr_reco_b 2e-6
  --dual_joint_weight_decay 1e-4
  --dual_joint_warmup_epochs 3
  --dual_joint_lambda_rank 0.2
  --dual_joint_rank_tau 0.05
  --dual_joint_lambda_anchor_a 0.02
  --dual_joint_lambda_anchor_b 0.02

  --report_target_tpr 0.50
  --step1_load_dir "${STEP1_LOAD_DIR}"
  --load_recoA_ckpt "${LOAD_RECO_A_CKPT}"
  --load_recoB_ckpt "${LOAD_RECO_B_CKPT}"
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-17 dual-reco dualview anti-overlap tagger-from-reco (weighted)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Train path: ${TRAIN_PATH} (weighted)"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "teacher_mode=anti_overlap, lambda=${TEACHER_ANTI_LAMBDA}, tau=${TEACHER_ANTI_TAU}, beta=${TEACHER_ANTI_BETA}, warmup=${TEACHER_ANTI_WARMUP_EPOCHS}"
echo "target_mode=offdrop, target_drop_prob_max=${OFFDROP_PROB_MAX}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
