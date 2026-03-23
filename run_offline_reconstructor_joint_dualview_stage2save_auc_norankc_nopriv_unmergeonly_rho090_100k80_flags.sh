#!/usr/bin/env bash
# Non-privileged unmerge-only variant:
# - No privileged merge/eff split targets.
# - Efficiency generation hard-disabled in script.
# - Added target uses conservative scaled true_added (default scale=0.90).
# - Merge/eff flag channels disabled.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_rho090_100k80_flags.sh

#SBATCH --job-name=nrivUO9
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=3-12:00:00
#SBATCH --output=offline_reconstructor_logs/1M_100_offline_reco_nopriv_uo_rho090_%j.out
#SBATCH --error=offline_reconstructor_logs/1M_100_offline_reco_nopriv_uo_rho090_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_1MJ100C}"
N_TRAIN_JETS="${N_TRAIN_JETS:-1000000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-0.90}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"
STAGEB_LAMBDA_RANK="${STAGEB_LAMBDA_RANK:-0.0}"
STAGEB_LAMBDA_CONS="${STAGEB_LAMBDA_CONS:-0.0}"
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

echo "Running nopriv-unmergeonly (scaled true_added target):"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric ${SELECTION_METRIC} --stageB_lambda_rank ${STAGEB_LAMBDA_RANK} --stageB_lambda_cons ${STAGEB_LAMBDA_CONS} --stageC_lr_dual ${STAGEC_LR_DUAL} --stageC_lr_reco ${STAGEC_LR_RECO} --lambda_reco ${LAMBDA_RECO} --lambda_cons ${LAMBDA_CONS} --added_target_scale ${ADDED_TARGET_SCALE} --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric "${SELECTION_METRIC}" \
  --stageB_lambda_rank "${STAGEB_LAMBDA_RANK}" \
  --stageB_lambda_cons "${STAGEB_LAMBDA_CONS}" \
  --stageC_lr_dual "${STAGEC_LR_DUAL}" \
  --stageC_lr_reco "${STAGEC_LR_RECO}" \
  --lambda_reco "${LAMBDA_RECO}" \
  --lambda_cons "${LAMBDA_CONS}" \
  --added_target_scale "${ADDED_TARGET_SCALE}" \
  --disable_final_kd \
  --device cuda
