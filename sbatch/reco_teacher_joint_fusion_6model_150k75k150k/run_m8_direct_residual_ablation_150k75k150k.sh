#!/usr/bin/env bash
#SBATCH --job-name=m8dir
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m8_direct_residual_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m8_direct_residual_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model8_direct_residual_ablation_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model8_direct_residual_ablation}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
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
  python reco_teacher_stageA_residual_hlt.py
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

  # Stage-A: strong teacher alignment (s09-style foundation)
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
  --stageA_lambda_delta 0.20
  --stageA_delta_tau 0.05
  --stageA_delta_lambda_fp 3.0
  --stageA_loss_norm_ema_decay 0.98
  --stageA_loss_norm_eps 1e-6
  --added_target_scale 0.90
  --reco_weight_threshold 0.03
  --reco_eval_batch_size 256

  # Residual head: residual-dominant objective (ablation)
  --residual_epochs 60
  --residual_patience 14
  --residual_lr 4e-4
  --residual_weight_decay 1e-4
  --residual_warmup_epochs 6
  --residual_lambda_res 1.5
  --residual_lambda_kd 0.6
  --residual_lambda_cls 0.3
  --residual_kd_temp 2.5
  --residual_select_metric fpr50
  --residual_alpha_grid 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0

  # Light joint unfreeze: weak reco anchor so optimization can follow residual objective
  --residual_joint_epochs 18
  --residual_joint_patience 10
  --residual_joint_lr_reco 4e-6
  --residual_joint_lr_head 1.5e-4
  --residual_joint_weight_decay 1e-4
  --residual_joint_warmup_epochs 4
  --residual_joint_lambda_reco_anchor 0.005

  --report_target_tpr 0.50
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-8 Direct-Residual Ablation (residual-dominant)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
