#!/usr/bin/env bash
# Non-privileged rho-split + split-again + teacher-guided reconstructor loss variant:
# - Stage A reconstructor objective: normalized KD + embedding + token + phys + budget-hinge.
# - Stage A selection: teacher-on-reco val AUC.
# - Stage C reconstructor term uses the same teacher-guided objective (scaled by --lambda_reco).
# - No merge/eff corrected-flag channels.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd_rho090_300k80k500k_noflags.sh

#SBATCH --job-name=nrivRSAKD
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_teacherkd_rho090_75k35k200k_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_teacherkd_rho090_75k35k200k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd_rho090_75k35k200k_100c_noflags_seed0}"
N_TRAIN_JETS="${N_TRAIN_JETS:-310000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-75000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-35000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-200000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-0.90}"
SEED="${SEED:-0}"

STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"

STAGEA_KD_TEMP="${STAGEA_KD_TEMP:-2.5}"
STAGEA_LAMBDA_KD="${STAGEA_LAMBDA_KD:-1.0}"
STAGEA_LAMBDA_EMB="${STAGEA_LAMBDA_EMB:-1.2}"
STAGEA_LAMBDA_TOK="${STAGEA_LAMBDA_TOK:-0.6}"
STAGEA_LAMBDA_PHYS="${STAGEA_LAMBDA_PHYS:-0.2}"
STAGEA_LAMBDA_BUDGET_HINGE="${STAGEA_LAMBDA_BUDGET_HINGE:-0.03}"
STAGEA_BUDGET_EPS="${STAGEA_BUDGET_EPS:-0.015}"
STAGEA_BUDGET_WEIGHT_FLOOR="${STAGEA_BUDGET_WEIGHT_FLOOR:-1e-4}"
STAGEA_TARGET_TPR="${STAGEA_TARGET_TPR:-0.50}"
STAGEA_LOSS_NORM_EMA_DECAY="${STAGEA_LOSS_NORM_EMA_DECAY:-0.98}"
STAGEA_LOSS_NORM_EPS="${STAGEA_LOSS_NORM_EPS:-1e-6}"
DISABLE_STAGEA_LOSS_NORMALIZATION="${DISABLE_STAGEA_LOSS_NORMALIZATION:-0}"
REPORT_TARGET_TPR="${REPORT_TARGET_TPR:-0.50}"
COMBO_WEIGHT_STEP="${COMBO_WEIGHT_STEP:-0.01}"
DIAG_MATCH_MAX_JETS="${DIAG_MATCH_MAX_JETS:-20000}"
DIAG_MATCH_SEED="${DIAG_MATCH_SEED:--1}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

EXTRA_STAGEA_NORM_FLAG=()
if [[ "${DISABLE_STAGEA_LOSS_NORMALIZATION}" == "1" ]]; then
  EXTRA_STAGEA_NORM_FLAG+=(--disable_stageA_loss_normalization)
fi

echo "Running nopriv-rhosplit + split-again + StageA/StageC teacher-guided reco loss:"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --seed ${SEED} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual ${STAGEC_LR_DUAL} --stageC_lr_reco ${STAGEC_LR_RECO} --lambda_reco ${LAMBDA_RECO} --lambda_cons ${LAMBDA_CONS} --added_target_scale ${ADDED_TARGET_SCALE} --stageA_kd_temp ${STAGEA_KD_TEMP} --stageA_lambda_kd ${STAGEA_LAMBDA_KD} --stageA_lambda_emb ${STAGEA_LAMBDA_EMB} --stageA_lambda_tok ${STAGEA_LAMBDA_TOK} --stageA_lambda_phys ${STAGEA_LAMBDA_PHYS} --stageA_lambda_budget_hinge ${STAGEA_LAMBDA_BUDGET_HINGE} --stageA_budget_eps ${STAGEA_BUDGET_EPS} --stageA_budget_weight_floor ${STAGEA_BUDGET_WEIGHT_FLOOR} --stageA_target_tpr ${STAGEA_TARGET_TPR} --stageA_loss_norm_ema_decay ${STAGEA_LOSS_NORM_EMA_DECAY} --stageA_loss_norm_eps ${STAGEA_LOSS_NORM_EPS} --report_target_tpr ${REPORT_TARGET_TPR} --combo_weight_step ${COMBO_WEIGHT_STEP} --diag_match_max_jets ${DIAG_MATCH_MAX_JETS} --diag_match_seed ${DIAG_MATCH_SEED} --disable_final_kd --device cuda ${EXTRA_STAGEA_NORM_FLAG[*]}"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --n_train_split "${N_TRAIN_SPLIT}" \
  --n_val_split "${N_VAL_SPLIT}" \
  --n_test_split "${N_TEST_SPLIT}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --stageC_lr_dual "${STAGEC_LR_DUAL}" \
  --stageC_lr_reco "${STAGEC_LR_RECO}" \
  --lambda_reco "${LAMBDA_RECO}" \
  --lambda_cons "${LAMBDA_CONS}" \
  --added_target_scale "${ADDED_TARGET_SCALE}" \
  --stageA_kd_temp "${STAGEA_KD_TEMP}" \
  --stageA_lambda_kd "${STAGEA_LAMBDA_KD}" \
  --stageA_lambda_emb "${STAGEA_LAMBDA_EMB}" \
  --stageA_lambda_tok "${STAGEA_LAMBDA_TOK}" \
  --stageA_lambda_phys "${STAGEA_LAMBDA_PHYS}" \
  --stageA_lambda_budget_hinge "${STAGEA_LAMBDA_BUDGET_HINGE}" \
  --stageA_budget_eps "${STAGEA_BUDGET_EPS}" \
  --stageA_budget_weight_floor "${STAGEA_BUDGET_WEIGHT_FLOOR}" \
  --stageA_target_tpr "${STAGEA_TARGET_TPR}" \
  --stageA_loss_norm_ema_decay "${STAGEA_LOSS_NORM_EMA_DECAY}" \
  --stageA_loss_norm_eps "${STAGEA_LOSS_NORM_EPS}" \
  --report_target_tpr "${REPORT_TARGET_TPR}" \
  --combo_weight_step "${COMBO_WEIGHT_STEP}" \
  --diag_match_max_jets "${DIAG_MATCH_MAX_JETS}" \
  --diag_match_seed "${DIAG_MATCH_SEED}" \
  "${EXTRA_STAGEA_NORM_FLAG[@]}" \
  --disable_final_kd \
  --device cuda
