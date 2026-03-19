#!/usr/bin/env bash
# Hard-route MoE variant for nopriv-rhosplit + split-again:
# - Stage A: two reconstructors, hard routed by HLT constituent count threshold.
# - Stage B: two dual taggers trained on full data with route-weighted BCE.
# - Stage C: hard-routed joint finetune per branch.
# - Test-time: hard routing and merged prediction.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_hardroute_moe_rho090_1MJ100C_noflags.sh

#SBATCH --job-name=hrMoE50k
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_hardroute_moe_rho090_50k20k100k_100c_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_nopriv_rhosplit_splitagain_hardroute_moe_rho090_50k20k100k_100c_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_rhosplit_splitagain_hardroute_moe_rho090_50k20k100k_100c_noflags}"
N_TRAIN_JETS="${N_TRAIN_JETS:-170000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-50000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-20000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-100000}"
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
ROUTE_THR="${ROUTE_THR:-26}"
STAGEB_ROUTE_WEIGHT="${STAGEB_ROUTE_WEIGHT:-5.0}"
ROUTE_BOUNDARY_BAND="${ROUTE_BOUNDARY_BAND:-2}"

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

echo "Running nopriv-rhosplit + split-again hardroute MoE:"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_hardroute_moe.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --seed ${SEED} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual ${STAGEC_LR_DUAL} --stageC_lr_reco ${STAGEC_LR_RECO} --lambda_reco ${LAMBDA_RECO} --lambda_cons ${LAMBDA_CONS} --added_target_scale ${ADDED_TARGET_SCALE} --route_hlt_count_thr ${ROUTE_THR} --stageB_route_weight ${STAGEB_ROUTE_WEIGHT} --route_boundary_band ${ROUTE_BOUNDARY_BAND} --disable_final_kd --device cuda"

python3 offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_hardroute_moe.py \
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
  --route_hlt_count_thr "${ROUTE_THR}" \
  --stageB_route_weight "${STAGEB_ROUTE_WEIGHT}" \
  --route_boundary_band "${ROUTE_BOUNDARY_BAND}" \
  --disable_final_kd \
  --device cuda
