#!/usr/bin/env bash
# Non-privileged rho-split + split-again variant:
# - No privileged merge/eff labels.
# - true_missing = offline_count - hlt_count.
# - Targets: merge=rho*true_missing, eff=(1-rho)*true_missing.
# - Efficiency generation is enabled.
# - Corrected view includes conservative budget-aware split-again expansion.
# - Merge/eff corrected-flag channels disabled.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_rho090_100k80_noflags.sh

#SBATCH --job-name=nrivRSA1M
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=2-12:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_rho090_1MJ100C_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_rho090_1MJ100C_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stage2save_auc_norankc_nopriv_rhosplit_splitagain_rho090_1MJ100C_noflags}"
N_TRAIN_JETS="${N_TRAIN_JETS:-1000000}"
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

echo "Running nopriv-rhosplit + split-again (merge=rho*missing, eff=(1-rho)*missing):"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --seed ${SEED} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual ${STAGEC_LR_DUAL} --stageC_lr_reco ${STAGEC_LR_RECO} --lambda_reco ${LAMBDA_RECO} --lambda_cons ${LAMBDA_CONS} --added_target_scale ${ADDED_TARGET_SCALE} --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
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
  --disable_final_kd \
  --device cuda
