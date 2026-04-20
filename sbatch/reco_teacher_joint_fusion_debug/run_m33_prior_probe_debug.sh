#!/usr/bin/env bash
#SBATCH --job-name=m33probe
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_debug/m33_prior_probe_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_debug/m33_prior_probe_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_debug

RUN_NAME="${RUN_NAME:-m33_prior_probe_debug_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_debug/m33_prior_probe}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-80}"

# Quick debug split: enough to catch manifold failures without huge runtime.
N_TRAIN_JETS="${N_TRAIN_JETS:-45000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-12000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-4000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-8000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

TEACHER_EPOCHS="${TEACHER_EPOCHS:-10}"
TEACHER_PATIENCE="${TEACHER_PATIENCE:-3}"
PRIOR_EPOCHS="${PRIOR_EPOCHS:-20}"
PRIOR_PATIENCE="${PRIOR_PATIENCE:-5}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-8}"
N_PRIOR_SAMPLES_PER_CLASS="${N_PRIOR_SAMPLES_PER_CLASS:-1000}"
ROUNDTRIP_EVAL_COUNT="${ROUNDTRIP_EVAL_COUNT:-1500}"

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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_prior_probe.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --teacher_epochs "${TEACHER_EPOCHS}"
  --teacher_patience "${TEACHER_PATIENCE}"
  --prior_epochs "${PRIOR_EPOCHS}"
  --prior_patience "${PRIOR_PATIENCE}"
  --critic_epochs "${CRITIC_EPOCHS}"
  --n_prior_samples_per_class "${N_PRIOR_SAMPLES_PER_CLASS}"
  --roundtrip_eval_count "${ROUNDTRIP_EVAL_COUNT}"
)

echo "============================================================"
echo "m33 prior probe (debug)"
echo "Run:   ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
