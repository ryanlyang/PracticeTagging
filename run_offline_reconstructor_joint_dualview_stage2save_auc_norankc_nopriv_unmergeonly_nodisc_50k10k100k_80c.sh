#!/usr/bin/env bash
# Full nopriv-unmergeonly pipeline WITHOUT discrepancy weighting.
# Matched to the discweighted runner for clean A/B comparison:
# - Same split counts: train=50k, val=10k, test=100k (total 160k loaded)
# - Same seed/setup defaults
# - Same Stage B/C LRs and lambda_reco/lambda_cons defaults
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_nodisc_50k10k100k_80c.sh

#SBATCH --job-name=uoNoDisc50
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_uo_nodisc_50k10k100k80_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_uo_nodisc_50k10k100k80_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_unmergeonly_nodisc_50k10k100k_80c}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
OFFSET_JETS="${OFFSET_JETS:-0}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"

# Total jets loaded for this run and exact split counts:
N_TRAIN_JETS="${N_TRAIN_JETS:-160000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-50000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-10000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-100000}"

ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-0.90}"
SEED="${SEED:-0}"

# Baseline staged training knobs (matched to discweighted runner defaults).
STAGEB_LR_DUAL="${STAGEB_LR_DUAL:-4e-4}"
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

cmd=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
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
  --stageB_lr_dual "${STAGEB_LR_DUAL}"
  --stageC_lr_dual "${STAGEC_LR_DUAL}"
  --stageC_lr_reco "${STAGEC_LR_RECO}"
  --lambda_reco "${LAMBDA_RECO}"
  --lambda_cons "${LAMBDA_CONS}"
  --added_target_scale "${ADDED_TARGET_SCALE}"
  --disable_final_kd
  --device cuda
)

echo "Running full nopriv-unmergeonly pipeline (no discrepancy weighting):"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

