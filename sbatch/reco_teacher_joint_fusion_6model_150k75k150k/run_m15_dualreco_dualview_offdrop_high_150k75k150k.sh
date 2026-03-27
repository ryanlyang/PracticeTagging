#!/usr/bin/env bash
#SBATCH --job-name=m15hig
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m15_dualreco_offdrop_high_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m15_dualreco_offdrop_high_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model15_dualreco_dualview_offdrop_high_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model15_dualreco_dualview_offdrop_high}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

OFFDROP_PROB_MAX="${OFFDROP_PROB_MAX:-0.70}"
RATIO_COUNT_UNDER_LAMBDA="${RATIO_COUNT_UNDER_LAMBDA:-1.0}"
RATIO_COUNT_OVER_LAMBDA="${RATIO_COUNT_OVER_LAMBDA:-0.25}"
RATIO_COUNT_MARGIN_BASE="${RATIO_COUNT_MARGIN_BASE:-2.0}"
RATIO_COUNT_MARGIN_SCALE="${RATIO_COUNT_MARGIN_SCALE:-6.0}"
RATIO_COUNT_GAMMA="${RATIO_COUNT_GAMMA:-0.70}"
RATIO_COUNT_OVER_FLOOR="${RATIO_COUNT_OVER_FLOOR:-0.05}"
RATIO_COUNT_EPS="${RATIO_COUNT_EPS:-0.015}"

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

mkdir -p "${SAVE_DIR}"

CMD=(
  python train_m9_dualreco_dualview_offdrop.py
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

  --teacher_use_offline_dropout
  --teacher_drop_prob_max "${OFFDROP_PROB_MAX}"
  --teacher_drop_warmup_epochs 20
  --teacher_drop_mode deterministic_bank
  --teacher_drop_num_banks 3
  --teacher_drop_bank_cycle_epochs 1
  --teacher_lambda_drop_cls 1.0
  --teacher_use_consistency
  --teacher_consistency_temp 2.0
  --teacher_lambda_consistency 0.2

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
  --target_drop_num_banks 3
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
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-15 HIGH dual-reco dualview (no residual head)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "offdrop_prob_max=${OFFDROP_PROB_MAX} (teacher + recoB target)"
echo "teacher_drop_mode=deterministic_bank, banks=3, bank_cycle_epochs=1"
echo "recoB ratio-count budget: under=${RATIO_COUNT_UNDER_LAMBDA} over=${RATIO_COUNT_OVER_LAMBDA} margin_base=${RATIO_COUNT_MARGIN_BASE} margin_scale=${RATIO_COUNT_MARGIN_SCALE} gamma=${RATIO_COUNT_GAMMA} floor=${RATIO_COUNT_OVER_FLOOR} eps=${RATIO_COUNT_EPS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
