#!/usr/bin/env bash
# Full nopriv-unmergeonly pipeline with discrepancy weighting:
# - Strong discrepancy weighting on reconstruction loss (Stage A + Stage C): lambda/max = 15/20
# - Light discrepancy weighting on dual/top-tagger BCE during Stage B
# - Custom exact splits: train=50k, val=10k, test=100k (total 160k loaded)
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_discweighted_50k10k100k_80c.sh

#SBATCH --job-name=uoDisc50
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_uo_disc_50k10k100k80_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_uo_disc_50k10k100k80_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_unmergeonly_discw_50k10k100k_80c}"
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

# Baseline staged training knobs (kept near current working defaults).
STAGEB_LR_DUAL="${STAGEB_LR_DUAL:-4e-4}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"

# Discrepancy weighting knobs:
# Strong reco weighting + light cls weighting.
DISC_WEIGHT_MODE="${DISC_WEIGHT_MODE:-smooth_delta}"
DISC_RECO_LAMBDA="${DISC_RECO_LAMBDA:-6.0}"
DISC_RECO_MAX_MULT="${DISC_RECO_MAX_MULT:-8.0}"
DISC_CLS_LAMBDA="${DISC_CLS_LAMBDA:-2.0}"
DISC_CLS_MAX_MULT="${DISC_CLS_MAX_MULT:-3.0}"
DISC_APPLY_CLS_STAGEC="${DISC_APPLY_CLS_STAGEC:-0}"
DISC_TARGET_TPR="${DISC_TARGET_TPR:-0.50}"
DISC_TAU="${DISC_TAU:-0.05}"
DISC_TEACHER_CONF_MIN="${DISC_TEACHER_CONF_MIN:-0.65}"
DISC_CORRECTNESS_TAU="${DISC_CORRECTNESS_TAU:-0.05}"
DISC_NO_MEAN_NORMALIZE="${DISC_NO_MEAN_NORMALIZE:-0}"

# Hard-correct-only gating by default.
DISC_DISABLE_TEACHER_HARD_CORRECT_GATE="${DISC_DISABLE_TEACHER_HARD_CORRECT_GATE:-0}"
DISC_DISABLE_TEACHER_CONF_GATE="${DISC_DISABLE_TEACHER_CONF_GATE:-0}"
DISC_DISABLE_TEACHER_BETTER_GATE="${DISC_DISABLE_TEACHER_BETTER_GATE:-0}"

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
  --disc_weight_enable
  --disc_weight_mode "${DISC_WEIGHT_MODE}"
  --disc_reco_lambda "${DISC_RECO_LAMBDA}"
  --disc_reco_max_mult "${DISC_RECO_MAX_MULT}"
  --disc_cls_lambda "${DISC_CLS_LAMBDA}"
  --disc_cls_max_mult "${DISC_CLS_MAX_MULT}"
  --disc_target_tpr "${DISC_TARGET_TPR}"
  --disc_tau "${DISC_TAU}"
  --disc_teacher_conf_min "${DISC_TEACHER_CONF_MIN}"
  --disc_correctness_tau "${DISC_CORRECTNESS_TAU}"
  --disable_final_kd
  --device cuda
)

if [[ "${DISC_NO_MEAN_NORMALIZE}" -eq 1 ]]; then
  cmd+=(--disc_no_mean_normalize)
fi
if [[ "${DISC_DISABLE_TEACHER_HARD_CORRECT_GATE}" -eq 1 ]]; then
  cmd+=(--disc_disable_teacher_hard_correct_gate)
fi
if [[ "${DISC_DISABLE_TEACHER_CONF_GATE}" -eq 1 ]]; then
  cmd+=(--disc_disable_teacher_conf_gate)
fi
if [[ "${DISC_DISABLE_TEACHER_BETTER_GATE}" -eq 1 ]]; then
  cmd+=(--disc_disable_teacher_better_gate)
fi
if [[ "${DISC_APPLY_CLS_STAGEC}" -eq 1 ]]; then
  cmd+=(--disc_apply_cls_stagec)
fi

echo "Running full discrepancy-weighted nopriv-unmergeonly pipeline:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"
