#!/usr/bin/env bash
# Non-privileged rho-split + split-again + teacher-guided reconstructor loss variant:
# - Stage A reconstructor objective: KD(teacher on reco view) + phys + budget-hinge.
# - Stage A selection: teacher-on-reco val AUC.
# - Stage C reconstructor term uses the same teacher-guided objective (scaled by --lambda_reco).
# - No merge/eff corrected-flag channels.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd_rho090_300k80k500k_noflags.sh

#SBATCH --job-name=nrivRSAKD
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_teacherkd_rho090_300k80k500k_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_teacherkd_rho090_300k80k500k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd_rho090_300k80k500k_100c_noflags_seed0}"
N_TRAIN_JETS="${N_TRAIN_JETS:-880000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-300000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-80000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-500000}"
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
STAGEA_LAMBDA_KD="${STAGEA_LAMBDA_KD:-5.0}"
STAGEA_LAMBDA_PHYS="${STAGEA_LAMBDA_PHYS:-0.05}"
STAGEA_LAMBDA_BUDGET_HINGE="${STAGEA_LAMBDA_BUDGET_HINGE:-1.0}"
STAGEA_BUDGET_EPS="${STAGEA_BUDGET_EPS:-0.015}"
STAGEA_BUDGET_WEIGHT_FLOOR="${STAGEA_BUDGET_WEIGHT_FLOOR:-1e-4}"
STAGEA_TARGET_TPR="${STAGEA_TARGET_TPR:-0.50}"

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

echo "Running nopriv-rhosplit + split-again + StageA/StageC teacher-guided reco loss:"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --seed ${SEED} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual ${STAGEC_LR_DUAL} --stageC_lr_reco ${STAGEC_LR_RECO} --lambda_reco ${LAMBDA_RECO} --lambda_cons ${LAMBDA_CONS} --added_target_scale ${ADDED_TARGET_SCALE} --stageA_kd_temp ${STAGEA_KD_TEMP} --stageA_lambda_kd ${STAGEA_LAMBDA_KD} --stageA_lambda_phys ${STAGEA_LAMBDA_PHYS} --stageA_lambda_budget_hinge ${STAGEA_LAMBDA_BUDGET_HINGE} --stageA_budget_eps ${STAGEA_BUDGET_EPS} --stageA_budget_weight_floor ${STAGEA_BUDGET_WEIGHT_FLOOR} --stageA_target_tpr ${STAGEA_TARGET_TPR} --disable_final_kd --device cuda"

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
  --stageA_lambda_phys "${STAGEA_LAMBDA_PHYS}" \
  --stageA_lambda_budget_hinge "${STAGEA_LAMBDA_BUDGET_HINGE}" \
  --stageA_budget_eps "${STAGEA_BUDGET_EPS}" \
  --stageA_budget_weight_floor "${STAGEA_BUDGET_WEIGHT_FLOOR}" \
  --stageA_target_tpr "${STAGEA_TARGET_TPR}" \
  --disable_final_kd \
  --device cuda
