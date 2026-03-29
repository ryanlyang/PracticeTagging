#!/usr/bin/env bash
#SBATCH --job-name=m2j000ut
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta000_unsmearcap_trust_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta000_unsmearcap_trust_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model2_joint_delta000_unsmearcap_trust_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta000_unsmearcap_trust}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

# Unsmear controls.
UNSMEAR_LOGPT_CAP="${UNSMEAR_LOGPT_CAP:-0.25}"
UNSMEAR_LOGE_CAP="${UNSMEAR_LOGE_CAP:-0.25}"
UNSMEAR_TRUST_LAMBDA="${UNSMEAR_TRUST_LAMBDA:-0.05}"
UNSMEAR_TRUST_TAU="${UNSMEAR_TRUST_TAU:-0.18}"

# Match the latest delta000 variants: keep these off.
LOSS_UNSELECTED_PENALTY="${LOSS_UNSELECTED_PENALTY:-0.00}"
LOSS_GEN_LOCAL_RADIUS="${LOSS_GEN_LOCAL_RADIUS:-0.00}"

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
export UNSMEAR_LOGPT_CAP="${UNSMEAR_LOGPT_CAP}"
export UNSMEAR_LOGE_CAP="${UNSMEAR_LOGE_CAP}"
export UNSMEAR_TRUST_LAMBDA="${UNSMEAR_TRUST_LAMBDA}"
export UNSMEAR_TRUST_TAU="${UNSMEAR_TRUST_TAU}"

mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_unsmearcap_trust.py
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
  --stageB_lambda_rank 0.0
  --stageB_lambda_cons 0.0
  --stageC_lr_dual 1e-5
  --stageC_lr_reco 5e-6
  --lambda_reco 0.4
  --lambda_cons 0.06
  --stageC_lambda_delta 0.00
  --stageC_delta_tau 0.05
  --stageC_delta_lambda_fp 3.0
  --stageC_delta_warmup_epochs 8
  --added_target_scale 0.90
  --loss_unselected_penalty "${LOSS_UNSELECTED_PENALTY}"
  --loss_gen_local_radius "${LOSS_GEN_LOCAL_RADIUS}"
  --save_fusion_scores
  --disable_final_kd
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-2 Joint (L_delta=0.00, unsmear hard-cap + trust penalty)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "Unsmear controls:"
echo "  UNSMEAR_LOGPT_CAP=${UNSMEAR_LOGPT_CAP}"
echo "  UNSMEAR_LOGE_CAP=${UNSMEAR_LOGE_CAP}"
echo "  UNSMEAR_TRUST_LAMBDA=${UNSMEAR_TRUST_LAMBDA}"
echo "  UNSMEAR_TRUST_TAU=${UNSMEAR_TRUST_TAU}"
echo "Reassign behavior: ORIGINAL (not geometry-only)"
echo "Reconstructor loss overrides:"
echo "  unselected_penalty=${LOSS_UNSELECTED_PENALTY}"
echo "  gen_local_radius=${LOSS_GEN_LOCAL_RADIUS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
