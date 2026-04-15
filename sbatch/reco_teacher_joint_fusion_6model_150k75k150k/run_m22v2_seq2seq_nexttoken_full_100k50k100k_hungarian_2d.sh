#!/usr/bin/env bash
#SBATCH --job-name=m22v2full
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m22v2_full_hungarian_2d_50k20k100k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m22v2_full_hungarian_2d_50k20k100k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model22v2_seq2seq_full_hungarian_2d_50k20k100k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model22_joint_seq2seq_nexttoken_v2}"
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

mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_seq2seq_nexttoken_v2.py
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
  --reco_epochs 180
  --reco_batch_size 128
  --reco_lr 1.8e-4
  --reco_patience 35
  --reco_min_epochs 60
  --reco_huber_delta 0.10
  --beam_size 6
  --beam_len_sigma 1.20
  --beam_temperature 0.85
  --use_coord_residual_param
  --enable_scale_sensitive_weighting
  --scale_pt_power 0.85
  --scale_weight_cap 6.0
  --angle_pt_power 1.0
  --loss_w_ar 1.0
  --loss_w_set 1.0
  --set_loss_mode hungarian
  --set_unmatched_penalty 0.35
  --loss_w_eos 0.20
  --loss_w_count 0.20
  --loss_w_ptr_entropy 0.0015
  --loss_w_angle 0.18
  --loss_w_jetpt 0.30
  --loss_w_jete 0.08
  --loss_w_4vec 0.10
  --physics_warmup_epochs 20
  --response_n_bins 20
  --response_min_count 200
  --save_fusion_scores
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-22v2 Seq2Seq Next-Token (Full Physics + Geometry, Hungarian, 2d)"
echo "Run:   ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "Jets:  n_train_jets=${N_TRAIN_JETS}, offset=${OFFSET_JETS}, max_constits=${MAX_CONSTITS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
