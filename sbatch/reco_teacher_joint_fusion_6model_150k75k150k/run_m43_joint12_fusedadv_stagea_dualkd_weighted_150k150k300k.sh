#!/usr/bin/env bash
#SBATCH --job-name=m43fa
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m43_joint12_fusedadv_stagea_dualkd_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m43_joint12_fusedadv_stagea_dualkd_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model43_joint12_fusedadv_stagea_dualkd_v2_weighted_150k150k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model43_joint12_fusedadv_stagea_dualkd_v2_weighted_150k150k300k}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
STEP1_LOAD_DIR="${STEP1_LOAD_DIR:-}"
FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k/fused_targets_train_val_test.npz}"
FUSED_TARGETS_KEY="${FUSED_TARGETS_KEY:-probs_fused_overall}"
FUSED_SPLIT_SCHEME="${FUSED_SPLIT_SCHEME:-train_val_test}"
STAGEA_FUSED_SOURCE_SPLITS_NPZ="${STAGEA_FUSED_SOURCE_SPLITS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"
STAGEA_FUSED_SOURCE_VAL_KEY="${STAGEA_FUSED_SOURCE_VAL_KEY:-val_idx}"
STAGEA_FUSED_SOURCE_TEST_KEY="${STAGEA_FUSED_SOURCE_TEST_KEY:-test_idx}"
STAGEA_FUSED_TRAIN_FROM="${STAGEA_FUSED_TRAIN_FROM:-fit}"
STAGEA_FUSED_VAL_FROM="${STAGEA_FUSED_VAL_FROM:-ref}"
JOINT_FUSED_SOURCE_SPLITS_NPZ="${JOINT_FUSED_SOURCE_SPLITS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"
JOINT_FUSED_SOURCE_VAL_KEY="${JOINT_FUSED_SOURCE_VAL_KEY:-val_idx}"
JOINT_FUSED_SOURCE_TEST_KEY="${JOINT_FUSED_SOURCE_TEST_KEY:-test_idx}"
JOINT_FUSED_TRAIN_FROM="${JOINT_FUSED_TRAIN_FROM:-fit}"
JOINT_FUSED_VAL_FROM="${JOINT_FUSED_VAL_FROM:-ref}"
STAGEA_FUSED_UNCERT_WEIGHT="${STAGEA_FUSED_UNCERT_WEIGHT:-0.50}"
STAGEA_FUSED_KD_W_MAX="${STAGEA_FUSED_KD_W_MAX:-4.00}"
JOINT_FUSED_KD_W_MAX="${JOINT_FUSED_KD_W_MAX:-3.00}"
STAGEC_JOINT_FUSED_KD_LAMBDA="${STAGEC_JOINT_FUSED_KD_LAMBDA:-0.28}"

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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd.py
  --train_path "${TRAIN_PATH}"
  --use_train_weights
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
  --stageA_loss_norm_ema_decay 0.98
  --stageA_loss_norm_eps 1e-6
  --stageA_fused_targets_npz "${FUSED_TARGETS_NPZ}"
  --stageA_fused_targets_key "${FUSED_TARGETS_KEY}"
  --stageA_fused_split_scheme "${FUSED_SPLIT_SCHEME}"
  --stageA_fused_source_splits_npz "${STAGEA_FUSED_SOURCE_SPLITS_NPZ}"
  --stageA_fused_source_val_key "${STAGEA_FUSED_SOURCE_VAL_KEY}"
  --stageA_fused_source_test_key "${STAGEA_FUSED_SOURCE_TEST_KEY}"
  --stageA_fused_train_from "${STAGEA_FUSED_TRAIN_FROM}"
  --stageA_fused_val_from "${STAGEA_FUSED_VAL_FROM}"
  --stageA_fused_adv_weight 2.0
  --stageA_fused_adv_power 1.0
  --stageA_fused_uncert_weight "${STAGEA_FUSED_UNCERT_WEIGHT}"
  --stageA_fused_adv_use_abs
  --stageA_fused_kd_w_min 0.30
  --stageA_fused_kd_w_max "${STAGEA_FUSED_KD_W_MAX}"
  --stageA_lambda_delta_aux 0.35
  --joint_fused_targets_npz "${FUSED_TARGETS_NPZ}"
  --joint_fused_targets_key "${FUSED_TARGETS_KEY}"
  --joint_fused_split_scheme "${FUSED_SPLIT_SCHEME}"
  --joint_fused_source_splits_npz "${JOINT_FUSED_SOURCE_SPLITS_NPZ}"
  --joint_fused_source_val_key "${JOINT_FUSED_SOURCE_VAL_KEY}"
  --joint_fused_source_test_key "${JOINT_FUSED_SOURCE_TEST_KEY}"
  --joint_fused_train_from "${JOINT_FUSED_TRAIN_FROM}"
  --joint_fused_val_from "${JOINT_FUSED_VAL_FROM}"
  --joint_fused_kd_temp 1.00
  --joint_fused_adv_weight 1.00
  --joint_fused_adv_power 1.00
  --joint_fused_uncert_weight 0.40
  --joint_fused_adv_use_abs
  --joint_fused_kd_w_min 0.30
  --joint_fused_kd_w_max "${JOINT_FUSED_KD_W_MAX}"
  --stageB_joint_fused_kd_lambda 0.12
  --stageC_joint_fused_kd_lambda "${STAGEC_JOINT_FUSED_KD_LAMBDA}"
  --stageB_lambda_rank 0.0
  --stageB_lambda_cons 0.0
  --stageC_lr_dual 1e-5
  --stageC_lr_reco 5e-6
  --lambda_reco 0.4
  --lambda_cons 0.06
  --added_target_scale 0.90
  --report_target_tpr 0.50
  --combo_weight_step 0.01
  --disable_final_kd
  --device "${DEVICE}"
)

if [[ -n "${STEP1_LOAD_DIR}" ]]; then
  CMD+=(--step1_load_dir "${STEP1_LOAD_DIR}")
fi

echo "============================================================"
echo "Model-43 Strategy (StageA fused-adv KD + StageB/C fused KD, v2)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Fused targets: ${FUSED_TARGETS_NPZ}"
echo "Source alignment: stageA(train=${STAGEA_FUSED_TRAIN_FROM},val=${STAGEA_FUSED_VAL_FROM}) joint(train=${JOINT_FUSED_TRAIN_FROM},val=${JOINT_FUSED_VAL_FROM})"
echo "Knobs: stageA_uncert=${STAGEA_FUSED_UNCERT_WEIGHT} stageA_kd_w_max=${STAGEA_FUSED_KD_W_MAX} joint_kd_w_max=${JOINT_FUSED_KD_W_MAX} stageC_joint_fused_kd_lambda=${STAGEC_JOINT_FUSED_KD_LAMBDA}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
