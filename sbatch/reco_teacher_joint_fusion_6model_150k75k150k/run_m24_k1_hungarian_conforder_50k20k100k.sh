#!/usr/bin/env bash
#SBATCH --job-name=m24k1
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m24_k1_50k20k100k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m24_k1_50k20k100k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model24_k1_hungarian_conforder_50k20k100k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model24_hungarian_conforder}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"

N_TRAIN_JETS="${N_TRAIN_JETS:-170000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-50000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-20000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-100000}"
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

CMD=(
  python offline_reconstructor_joint_dualview_seq2seq_nexttoken_m24_conforder.py
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
  --reco_epochs 150
  --reco_batch_size 112
  --reco_lr 1.8e-4
  --reco_patience 30
  --reco_min_epochs 45
  --reco_huber_delta 0.12
  --beam_size 4
  --beam_len_sigma 1.35
  --beam_temperature 0.9
  --num_hypotheses 1
  --joint_epochs 0
  --set_loss_mode hungarian
  --ar_use_hungarian_target
  --loss_w_ar 1.0
  --loss_w_set 1.0
  --loss_w_eos 0.20
  --loss_w_count 0.20
  --loss_w_ptr_entropy 0.002
  --loss_w_conf_rank 0.20
  --loss_w_conf_prefix 0.12
  --conf_margin 0.06
  --conf_prefix_tau 16.0
  --loss_w_jetpt 0.08
  --loss_w_4vec 0.03
  --loss_w_jete 0.00
  --loss_w_angle 0.00
  --physics_warmup_epochs 12
  --save_fusion_scores
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-24 K=1 Hungarian + Confidence-Order"
echo "Run:   ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
