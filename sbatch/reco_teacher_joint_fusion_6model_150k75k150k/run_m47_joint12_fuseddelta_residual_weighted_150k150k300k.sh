#!/usr/bin/env bash
#SBATCH --job-name=m47fd
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m47_joint12_fuseddelta_residual_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m47_joint12_fuseddelta_residual_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model47_joint12_fuseddelta_residual_weighted_150k150k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model47_joint12_fuseddelta_residual_weighted_150k150k300k}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k/fused_targets_train_val_test.npz}"
FUSED_TARGETS_KEY="${FUSED_TARGETS_KEY:-probs_fused_overall}"
FUSED_SPLIT_SCHEME="${FUSED_SPLIT_SCHEME:-train_val_test}"
FUSED_SOURCE_SPLITS_NPZ="${FUSED_SOURCE_SPLITS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"
RESIDUAL_TRAIN_FROM="${RESIDUAL_TRAIN_FROM:-fit}"
RESIDUAL_VAL_FROM="${RESIDUAL_VAL_FROM:-ref}"
RESIDUAL_TEST_FROM="${RESIDUAL_TEST_FROM:-source_test}"

N_TRAIN_JETS="${N_TRAIN_JETS:-600000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-150000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

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
  python reco_teacher_stageA_residual_fuseddelta.py
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

  --stageA_epochs 90
  --stageA_patience 18
  --stageA_kd_temp 2.5
  --stageA_lambda_kd 1.0
  --stageA_lambda_emb 1.2
  --stageA_lambda_tok 0.6
  --stageA_lambda_phys 0.2
  --stageA_lambda_budget_hinge 0.03
  --stageA_budget_eps 0.015
  --stageA_budget_weight_floor 1e-4
  --stageA_target_tpr 0.50
  --stageA_lambda_delta 0.15
  --stageA_delta_tau 0.05
  --stageA_delta_lambda_fp 3.0
  --stageA_loss_norm_ema_decay 0.98
  --stageA_loss_norm_eps 1e-6
  --added_target_scale 0.90

  --fused_targets_npz "${FUSED_TARGETS_NPZ}"
  --fused_targets_key "${FUSED_TARGETS_KEY}"
  --fused_split_scheme "${FUSED_SPLIT_SCHEME}"
  --fused_source_splits_npz "${FUSED_SOURCE_SPLITS_NPZ}"
  --fused_source_val_key val_idx
  --fused_source_test_key test_idx
  --residual_train_from "${RESIDUAL_TRAIN_FROM}"
  --residual_val_from "${RESIDUAL_VAL_FROM}"
  --residual_test_from "${RESIDUAL_TEST_FROM}"
  --fused_target_sanity_min_auc 0.75

  --reco_weight_threshold 0.03
  --reco_eval_batch_size 256
  --residual_epochs 45
  --residual_patience 12
  --residual_lr 3e-4
  --residual_weight_decay 1e-4
  --residual_warmup_epochs 5
  --residual_lambda_res 1.0
  --residual_lambda_kd 0.2
  --residual_lambda_cls 0.1
  --residual_kd_temp 2.5
  --residual_select_metric fpr50
  --residual_alpha_grid 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0

  --residual_joint_epochs 12
  --residual_joint_patience 10
  --residual_joint_lr_reco 2e-6
  --residual_joint_lr_head 1e-4
  --residual_joint_weight_decay 1e-4
  --residual_joint_warmup_epochs 4
  --residual_joint_lambda_reco_anchor 0.02

  --report_target_tpr 0.50
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-47 Joint12 fused-delta residual (weighted)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Fused targets: ${FUSED_TARGETS_NPZ}"
echo "Residual split mapping: train_from=${RESIDUAL_TRAIN_FROM}, val_from=${RESIDUAL_VAL_FROM}, test_from=${RESIDUAL_TEST_FROM}"
echo "Split (core): train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
