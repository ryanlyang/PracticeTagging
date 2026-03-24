#!/usr/bin/env bash
#SBATCH --job-name=m4c300
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m4_corrected_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m4_corrected_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model4_recoteacher_s01_corrected_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model4_recoteacher_s01_corrected}"
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
  python reco_teacher_stageA_only_delta_curriculum.py
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
  --stageA_lambda_delta 0.00
  --stageA_delta_tau 0.05
  --stageA_delta_lambda_fp 3.0
  --stageA_loss_norm_ema_decay 0.98
  --stageA_loss_norm_eps 1e-6
  --added_target_scale 0.90
  --report_target_tpr 0.50
  --combo_weight_step 0.01
  --train_corrected_only_post_stageA
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-4 StageA(s01) + Corrected-only tagger"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
