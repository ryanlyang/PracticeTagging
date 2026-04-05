#!/usr/bin/env bash
#SBATCH --job-name=m2jfht
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_fulltrain_prog_unfreeze_hungarian_tokenaux_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_fulltrain_prog_unfreeze_hungarian_tokenaux_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model2_joint_delta005_fulltrain_prog_unfreeze_hungarian_tokenaux_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_fulltrain_prog_unfreeze_hungarian_tokenaux}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

# Hungarian token auxiliary knobs.
HUNGARIAN_TOKENAUX_FORCE_SET="${HUNGARIAN_TOKENAUX_FORCE_SET:-1}"
HUNGARIAN_TOKEN_AUX_LAMBDA="${HUNGARIAN_TOKEN_AUX_LAMBDA:-0.15}"
HUNGARIAN_TOKEN_AUX_W_LOGPT="${HUNGARIAN_TOKEN_AUX_W_LOGPT:-1.0}"
HUNGARIAN_TOKEN_AUX_W_ETA="${HUNGARIAN_TOKEN_AUX_W_ETA:-0.6}"
HUNGARIAN_TOKEN_AUX_W_PHI="${HUNGARIAN_TOKEN_AUX_W_PHI:-0.6}"
HUNGARIAN_TOKEN_AUX_W_LOGE="${HUNGARIAN_TOKEN_AUX_W_LOGE:-0.25}"
HUNGARIAN_TOKEN_AUX_BETA="${HUNGARIAN_TOKEN_AUX_BETA:-0.08}"
HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT="${HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT:-0.05}"

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

export HUNGARIAN_TOKENAUX_FORCE_SET
export HUNGARIAN_TOKEN_AUX_LAMBDA
export HUNGARIAN_TOKEN_AUX_W_LOGPT
export HUNGARIAN_TOKEN_AUX_W_ETA
export HUNGARIAN_TOKEN_AUX_W_PHI
export HUNGARIAN_TOKEN_AUX_W_LOGE
export HUNGARIAN_TOKEN_AUX_BETA
export HUNGARIAN_TOKEN_AUX_MIN_MATCH_WEIGHT

mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_hungarian_tokenaux.py
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
  --loss_set_mode hungarian
  --stageC_progressive_unfreeze
  --stageC_unfreeze_phase1_epochs 3
  --stageC_unfreeze_phase2_epochs 7
  --stageC_unfreeze_last_n_encoder_layers 2
  --stageC_lambda_param_anchor 0.02
  --stageC_lambda_output_anchor 0.02
  --stageC_anchor_decay 0.97
  --stageC_lr_dual 1e-5
  --stageC_lr_reco 5e-6
  --lambda_reco 0.4
  --lambda_cons 0.06
  --loss_unselected_penalty 0.0
  --loss_gen_local_radius 0.0
  --stageC_lambda_delta 0.05
  --stageC_delta_tau 0.05
  --stageC_delta_lambda_fp 3.0
  --stageC_delta_warmup_epochs 8
  --added_target_scale 0.90
  --save_fusion_scores
  --disable_final_kd
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-2 full A/B/C training (Hungarian + matched-token auxiliary)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Hungarian forced set mode: ${HUNGARIAN_TOKENAUX_FORCE_SET}"
echo "Token-aux lambda:          ${HUNGARIAN_TOKEN_AUX_LAMBDA}"
echo "Token-aux weights:         logpt=${HUNGARIAN_TOKEN_AUX_W_LOGPT}, eta=${HUNGARIAN_TOKEN_AUX_W_ETA}, phi=${HUNGARIAN_TOKEN_AUX_W_PHI}, logE=${HUNGARIAN_TOKEN_AUX_W_LOGE}"
echo "Token-aux huber beta:      ${HUNGARIAN_TOKEN_AUX_BETA}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
