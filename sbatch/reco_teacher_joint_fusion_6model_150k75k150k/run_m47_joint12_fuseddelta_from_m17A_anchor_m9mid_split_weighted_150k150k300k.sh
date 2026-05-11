#!/usr/bin/env bash
#SBATCH --job-name=m47m9a
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m47_joint12_fuseddelta_from_m17A_anchor_m9mid_split_weighted_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m47_joint12_fuseddelta_from_m17A_anchor_m9mid_split_weighted_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model47_from_m17A_anchor_m9mid_split_weighted_150k150k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model47_from_m17A_anchor_m9mid_split_weighted_150k150k300k}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"
STAGEA_LOAD_CKPT="${STAGEA_LOAD_CKPT:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model17_dualreco_dualview_antioverlap_weighted_150k150k300k/model17_dualreco_dualview_antioverlap_weighted_150k150k300k_seed0/offline_reconstructor_A_stageA.pt}"

FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k_split_m47/fused_targets_train_val_test.npz}"
FUSED_TARGETS_KEY="${FUSED_TARGETS_KEY:-probs_fused_overall}"
FUSED_SPLIT_SCHEME="${FUSED_SPLIT_SCHEME:-train_val_test}"
FUSED_SOURCE_SPLITS_NPZ="${FUSED_SOURCE_SPLITS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"
RESIDUAL_TRAIN_FROM="${RESIDUAL_TRAIN_FROM:-fit}"
RESIDUAL_VAL_FROM="${RESIDUAL_VAL_FROM:-ref}"
RESIDUAL_TEST_FROM="${RESIDUAL_TEST_FROM:-source_test}"
RESIDUAL_SELECT_METRIC="${RESIDUAL_SELECT_METRIC:-auc}"
COMBO_WEIGHT_STEP="${COMBO_WEIGHT_STEP:-0.01}"

ANCHOR_LOGIT_SOURCE="${ANCHOR_LOGIT_SOURCE:-reco_teacher}"
ANCHOR_RECO_CKPT="${ANCHOR_RECO_CKPT:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model9_stageA_residual_hlt_offdrop_mid_weighted_150k150k300k/model9_stageA_residual_hlt_offdrop_mid_weighted_150k150k300k_seed0/offline_reconstructor_stageA.pt}"
ANCHOR_RECO_NON_STRICT="${ANCHOR_RECO_NON_STRICT:-0}"
ANCHOR_TEACHER_SOURCE="${ANCHOR_TEACHER_SOURCE:-teacher}"
ANCHOR_WEIGHT_THRESHOLD="${ANCHOR_WEIGHT_THRESHOLD:-0.03}"

N_TRAIN_JETS="${N_TRAIN_JETS:-600000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-150000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

RESIDUAL_JOINT_EPOCHS="${RESIDUAL_JOINT_EPOCHS:-36}"
RESIDUAL_JOINT_PATIENCE="${RESIDUAL_JOINT_PATIENCE:-14}"
RESIDUAL_JOINT_LR_RECO="${RESIDUAL_JOINT_LR_RECO:-2e-6}"
RESIDUAL_JOINT_LR_HEAD="${RESIDUAL_JOINT_LR_HEAD:-1e-4}"
RESIDUAL_JOINT_WEIGHT_DECAY="${RESIDUAL_JOINT_WEIGHT_DECAY:-1e-4}"
RESIDUAL_JOINT_WARMUP_EPOCHS="${RESIDUAL_JOINT_WARMUP_EPOCHS:-4}"
RESIDUAL_JOINT_LAMBDA_RECO_ANCHOR="${RESIDUAL_JOINT_LAMBDA_RECO_ANCHOR:-0.02}"

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

  --anchor_logit_source "${ANCHOR_LOGIT_SOURCE}"
  --anchor_reco_ckpt "${ANCHOR_RECO_CKPT}"
  --anchor_teacher_source "${ANCHOR_TEACHER_SOURCE}"
  --anchor_weight_threshold "${ANCHOR_WEIGHT_THRESHOLD}"

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
  --residual_select_metric "${RESIDUAL_SELECT_METRIC}"
  --residual_alpha_grid 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0

  --residual_joint_epochs "${RESIDUAL_JOINT_EPOCHS}"
  --residual_joint_patience "${RESIDUAL_JOINT_PATIENCE}"
  --residual_joint_lr_reco "${RESIDUAL_JOINT_LR_RECO}"
  --residual_joint_lr_head "${RESIDUAL_JOINT_LR_HEAD}"
  --residual_joint_weight_decay "${RESIDUAL_JOINT_WEIGHT_DECAY}"
  --residual_joint_warmup_epochs "${RESIDUAL_JOINT_WARMUP_EPOCHS}"
  --residual_joint_lambda_reco_anchor "${RESIDUAL_JOINT_LAMBDA_RECO_ANCHOR}"

  --report_target_tpr 0.50
  --combo_weight_step "${COMBO_WEIGHT_STEP}"
  --device "${DEVICE}"
)

if [[ -n "${STAGEA_LOAD_CKPT}" ]]; then
  CMD+=(--stageA_load_ckpt "${STAGEA_LOAD_CKPT}")
fi
if [[ "${ANCHOR_RECO_NON_STRICT}" == "1" ]]; then
  CMD+=(--anchor_reco_non_strict)
fi

echo "============================================================"
echo "Model-47 Joint12 fused-delta residual (m17 StageA + offdrop_mid anchor reco)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "StageA load ckpt: ${STAGEA_LOAD_CKPT:-<none>}"
echo "Anchor source: ${ANCHOR_LOGIT_SOURCE}"
echo "Anchor reco ckpt: ${ANCHOR_RECO_CKPT}"
echo "Anchor scorer: ${ANCHOR_TEACHER_SOURCE}"
echo "Fused targets: ${FUSED_TARGETS_NPZ}"
echo "Residual split mapping: train_from=${RESIDUAL_TRAIN_FROM}, val_from=${RESIDUAL_VAL_FROM}, test_from=${RESIDUAL_TEST_FROM}"
echo "Residual selection metric: ${RESIDUAL_SELECT_METRIC}"
echo "Combo weight step: ${COMBO_WEIGHT_STEP}"
echo "Joint finetune: epochs=${RESIDUAL_JOINT_EPOCHS}, patience=${RESIDUAL_JOINT_PATIENCE}"
echo "Split (core): train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
