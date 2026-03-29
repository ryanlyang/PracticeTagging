#!/usr/bin/env bash
#SBATCH --job-name=m2j000rg
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta000_reassigngeom_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta000_reassigngeom_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model2_joint_delta000_reassigngeom_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta000_reassigngeom}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

# Reconstructor loss settings (explicitly pinned to m2 delta000 defaults).
LOSS_SET_MODE="${LOSS_SET_MODE:-chamfer}"
LOSS_W_PHYS="${LOSS_W_PHYS:-0.35}"
LOSS_W_PT_RATIO="${LOSS_W_PT_RATIO:-0.70}"
LOSS_W_M_RATIO="${LOSS_W_M_RATIO:-0.00}"
LOSS_W_E_RATIO="${LOSS_W_E_RATIO:-0.35}"
LOSS_W_RADIAL_PROFILE="${LOSS_W_RADIAL_PROFILE:-0.00}"
LOSS_W_BUDGET="${LOSS_W_BUDGET:-0.65}"
LOSS_W_SPARSE="${LOSS_W_SPARSE:-0.02}"
LOSS_W_LOCAL="${LOSS_W_LOCAL:-0.03}"
LOSS_W_FP_MASS="${LOSS_W_FP_MASS:-0.00}"
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

mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_reassigngeom.py
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
  --loss_set_mode "${LOSS_SET_MODE}"
  --loss_w_phys "${LOSS_W_PHYS}"
  --loss_w_pt_ratio "${LOSS_W_PT_RATIO}"
  --loss_w_m_ratio "${LOSS_W_M_RATIO}"
  --loss_w_e_ratio "${LOSS_W_E_RATIO}"
  --loss_w_radial_profile "${LOSS_W_RADIAL_PROFILE}"
  --loss_w_budget "${LOSS_W_BUDGET}"
  --loss_w_sparse "${LOSS_W_SPARSE}"
  --loss_w_local "${LOSS_W_LOCAL}"
  --loss_w_fp_mass "${LOSS_W_FP_MASS}"
  --loss_unselected_penalty "${LOSS_UNSELECTED_PENALTY}"
  --loss_gen_local_radius "${LOSS_GEN_LOCAL_RADIUS}"
  --save_fusion_scores
  --disable_final_kd
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-2 Joint (L_delta=0.00 off, reassign=geometry-only)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "Reconstructor loss config:"
echo "  set_mode=${LOSS_SET_MODE}"
echo "  w_set=1.00"
echo "  w_phys=${LOSS_W_PHYS}"
echo "  w_pt_ratio=${LOSS_W_PT_RATIO}"
echo "  w_m_ratio=${LOSS_W_M_RATIO}"
echo "  w_e_ratio=${LOSS_W_E_RATIO}"
echo "  w_radial_profile=${LOSS_W_RADIAL_PROFILE}"
echo "  w_budget=${LOSS_W_BUDGET}"
echo "  w_sparse=${LOSS_W_SPARSE}"
echo "  w_local=${LOSS_W_LOCAL}"
echo "  w_fp_mass=${LOSS_W_FP_MASS}"
echo "  unselected_penalty=${LOSS_UNSELECTED_PENALTY}"
echo "  gen_local_radius=${LOSS_GEN_LOCAL_RADIUS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
