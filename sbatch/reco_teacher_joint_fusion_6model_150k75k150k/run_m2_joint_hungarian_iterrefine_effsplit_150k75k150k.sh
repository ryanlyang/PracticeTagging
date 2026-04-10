#!/usr/bin/env bash
#SBATCH --job-name=m2iter
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=22:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_iterrefine_effsplit_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_iterrefine_effsplit_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

HLT_EDIT_MODE="${HLT_EDIT_MODE:-fixed}"   # fixed | tiny_edit
WARM_START_MODE="${WARM_START_MODE:-cold}"  # cold | warm
WARM_START_RUN_DIR="${WARM_START_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_hungarian_nophys_noratio/model2_joint_hungarian_nophys_noratio_150k75k150k_seed0}"
WARM_START_RECO_CKPT="${WARM_START_RECO_CKPT:-}"
WARM_START_DUAL_CKPT="${WARM_START_DUAL_CKPT:-}"
RUN_NAME="${RUN_NAME:-model2_joint_hungarian_iterrefine_effsplit_${HLT_EDIT_MODE}_${WARM_START_MODE}_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_hungarian_iterrefine_effsplit}"
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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_iterrefine_effsplit.py
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
  --stageC_lambda_delta 0.05
  --stageC_delta_tau 0.05
  --stageC_delta_lambda_fp 3.0
  --stageC_delta_warmup_epochs 8
  --added_target_scale 1.0
  --added_target_merge_frac 0.85
  --iter_steps_early 1
  --iter_steps_mid 3
  --iter_steps_late 5
  --iter_cap_logpt 0.05
  --iter_cap_eta 0.02
  --iter_cap_phi 0.02
  --iter_cap_loge 0.05
  --iter_hlt_edit_mode "${HLT_EDIT_MODE}"
  --iter_hlt_edit_scale 0.25
  --warm_start_mode "${WARM_START_MODE}"
  --warm_start_run_dir "${WARM_START_RUN_DIR}"
  --loss_set_mode hungarian
  --loss_w_phys 0.0
  --loss_w_pt_ratio 0.0
  --loss_w_e_ratio 0.0
  --loss_w_local 0.05
  --save_fusion_scores
  --disable_final_kd
  --device "${DEVICE}"
)

if [[ -n "${WARM_START_RECO_CKPT}" ]]; then
  CMD+=(--warm_start_reco_ckpt "${WARM_START_RECO_CKPT}")
fi
if [[ -n "${WARM_START_DUAL_CKPT}" ]]; then
  CMD+=(--warm_start_dual_ckpt "${WARM_START_DUAL_CKPT}")
fi

echo "============================================================"
echo "Model-2 Joint IterRefine+EffSplit (Dual-view)"
echo "SetLoss=Hungarian, L_phys=L_pt_ratio=L_e_ratio=0, target_split=0.85/0.15, target_scale=1.0"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "HLT edit mode: ${HLT_EDIT_MODE}"
echo "Warm start mode: ${WARM_START_MODE}"
echo "Warm start run dir: ${WARM_START_RUN_DIR}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
