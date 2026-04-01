#!/usr/bin/env bash
#SBATCH --job-name=m2jftp
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_fulltrain_prog_unfreeze_trueposterior_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_fulltrain_prog_unfreeze_trueposterior_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model2_joint_delta005_fulltrain_prog_unfreeze_trueposterior_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_fulltrain_prog_unfreeze_trueposterior}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

# True posterior controls (conservative defaults)
POSTERIOR_K="${POSTERIOR_K:-3}"
POSTERIOR_MODE_DIM="${POSTERIOR_MODE_DIM:-32}"
POSTERIOR_TOKEN_HIDDEN="${POSTERIOR_TOKEN_HIDDEN:-192}"
POSTERIOR_CTX_HIDDEN="${POSTERIOR_CTX_HIDDEN:-96}"

POSTERIOR_PT_SHIFT="${POSTERIOR_PT_SHIFT:-0.35}"
POSTERIOR_E_SHIFT="${POSTERIOR_E_SHIFT:-0.35}"
POSTERIOR_ETA_SHIFT="${POSTERIOR_ETA_SHIFT:-0.30}"
POSTERIOR_PHI_SHIFT="${POSTERIOR_PHI_SHIFT:-0.30}"
POSTERIOR_W_SHIFT="${POSTERIOR_W_SHIFT:-1.00}"
POSTERIOR_W_RENORM_MIN="${POSTERIOR_W_RENORM_MIN:-0.50}"
POSTERIOR_W_RENORM_MAX="${POSTERIOR_W_RENORM_MAX:-2.00}"
POSTERIOR_ASSIGN_TAU="${POSTERIOR_ASSIGN_TAU:-0.30}"

POSTERIOR_W_CAL_STAGEA="${POSTERIOR_W_CAL_STAGEA:-0.03}"
POSTERIOR_W_DIV_STAGEA="${POSTERIOR_W_DIV_STAGEA:-0.02}"
POSTERIOR_W_ENT_STAGEA="${POSTERIOR_W_ENT_STAGEA:-0.01}"
POSTERIOR_W_CAL_JOINT="${POSTERIOR_W_CAL_JOINT:-0.00}"
POSTERIOR_W_DIV_JOINT="${POSTERIOR_W_DIV_JOINT:-0.00}"
POSTERIOR_W_ENT_JOINT="${POSTERIOR_W_ENT_JOINT:-0.00}"
POSTERIOR_ENT_TARGET="${POSTERIOR_ENT_TARGET:-0.35}"

POSTERIOR_GATE_HIDDEN="${POSTERIOR_GATE_HIDDEN:-48}"
POSTERIOR_GATE_DROPOUT="${POSTERIOR_GATE_DROPOUT:-0.05}"
POSTERIOR_GATE_PRIOR_SCALE="${POSTERIOR_GATE_PRIOR_SCALE:-0.70}"
POSTERIOR_ZERO_EFF_BRANCH="${POSTERIOR_ZERO_EFF_BRANCH:-1}"

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

export POSTERIOR_K
export POSTERIOR_MODE_DIM
export POSTERIOR_TOKEN_HIDDEN
export POSTERIOR_CTX_HIDDEN
export POSTERIOR_PT_SHIFT
export POSTERIOR_E_SHIFT
export POSTERIOR_ETA_SHIFT
export POSTERIOR_PHI_SHIFT
export POSTERIOR_W_SHIFT
export POSTERIOR_W_RENORM_MIN
export POSTERIOR_W_RENORM_MAX
export POSTERIOR_ASSIGN_TAU
export POSTERIOR_W_CAL_STAGEA
export POSTERIOR_W_DIV_STAGEA
export POSTERIOR_W_ENT_STAGEA
export POSTERIOR_W_CAL_JOINT
export POSTERIOR_W_DIV_JOINT
export POSTERIOR_W_ENT_JOINT
export POSTERIOR_ENT_TARGET
export POSTERIOR_GATE_HIDDEN
export POSTERIOR_GATE_DROPOUT
export POSTERIOR_GATE_PRIOR_SCALE
export POSTERIOR_ZERO_EFF_BRANCH

mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_trueposterior.py
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
echo "Model-2 full A/B/C training (TRUE POSTERIOR reconstructor + learned posterior aggregation)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Posterior K=${POSTERIOR_K}, mode_dim=${POSTERIOR_MODE_DIM}, token_hidden=${POSTERIOR_TOKEN_HIDDEN}"
echo "Assign tau=${POSTERIOR_ASSIGN_TAU}, stageA cal/div/ent=${POSTERIOR_W_CAL_STAGEA}/${POSTERIOR_W_DIV_STAGEA}/${POSTERIOR_W_ENT_STAGEA}"
echo "Joint cal/div/ent=${POSTERIOR_W_CAL_JOINT}/${POSTERIOR_W_DIV_JOINT}/${POSTERIOR_W_ENT_JOINT}"
echo "Gate hidden/dropout/prior=${POSTERIOR_GATE_HIDDEN}/${POSTERIOR_GATE_DROPOUT}/${POSTERIOR_GATE_PRIOR_SCALE}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"

