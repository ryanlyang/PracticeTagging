#!/usr/bin/env bash
#SBATCH --job-name=m33dsp
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_debug/m33_dhard_dsoft_probe_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_debug/m33_dhard_dsoft_probe_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_debug

RUN_NAME="${RUN_NAME:-m33_dhard_dsoft_probe_debug_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_debug/m33_dhard_dsoft_probe}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-80}"
RANK_EVAL_BATCH_SIZE="${RANK_EVAL_BATCH_SIZE:-256}"

N_TRAIN_JETS="${N_TRAIN_JETS:-70000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-20000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-8000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-12000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m33_dhard_dsoft_probe.py
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
  --degrader_epochs 32
  --degrader_patience 6
  --det_eval_count 2500
  --pert_eval_count 2000
  --cand_eval_count 1400
  --cand_per_jet 24
  --rank_top_m 8
  --rank_eval_batch_size "${RANK_EVAL_BATCH_SIZE}"
  --refine_eval_jets 350
  --refine_selected_k 6
  --refine_steps 6
  --refine_lr 0.03
)

echo "============================================================"
echo "m33 D_hard / D_soft probe (debug)"
echo "Run:   ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
