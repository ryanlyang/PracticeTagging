#!/usr/bin/env bash
#SBATCH --job-name=sA01bal
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=offline_reconstructor_logs/stageA_teacher_auc_sweep10/s01_balanced_%j.out
#SBATCH --error=offline_reconstructor_logs/stageA_teacher_auc_sweep10/s01_balanced_%j.err

set -euo pipefail
mkdir -p offline_reconstructor_logs/stageA_teacher_auc_sweep10

export RUN_NAME="joint_stageA_sweep10_s01_balanced_seed0"
export SAVE_DIR="checkpoints/offline_reconstructor_joint_stageA_teacher_auc_sweep10"
export N_TRAIN_JETS=1100000
export N_TRAIN_SPLIT=500000
export N_VAL_SPLIT=100000
export N_TEST_SPLIT=500000
export OFFSET_JETS=0
export MAX_CONSTITS=100
export NUM_WORKERS=6
export SEED=0
export DEVICE=cuda
export ANALYZE_DEVICE=cuda
export ANALYZE_NUM_WORKERS=1
export REPORT_TARGET_TPR=0.50
export COMBO_WEIGHT_STEP=0.01

export STAGEA_KD_TEMP=2.5
export STAGEA_LAMBDA_KD=1.0
export STAGEA_LAMBDA_EMB=1.2
export STAGEA_LAMBDA_TOK=0.6
export STAGEA_LAMBDA_PHYS=0.2
export STAGEA_LAMBDA_BUDGET_HINGE=0.03

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SCRIPT="${SUBMIT_DIR}/sbatch/stageA_teacher_auc_sweep10/run_stageA_sweep_common.sh"
if [[ ! -f "${COMMON_SCRIPT}" ]]; then
  echo "ERROR: common sweep script not found: ${COMMON_SCRIPT}" >&2
  exit 1
fi
bash "${COMMON_SCRIPT}"
