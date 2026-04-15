#!/usr/bin/env bash
#SBATCH --job-name=m23k6reco
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m23_k6_reco_50k20k100k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m23_k6_reco_50k20k100k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model23_multihyp_k6_reco_50k20k100k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model23_multihyp_k6}"
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
  python offline_reconstructor_joint_dualview_seq2seq_nexttoken_m23_multihyp.py
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
  --reco_epochs 140
  --reco_batch_size 96
  --reco_lr 1.8e-4
  --reco_patience 28
  --reco_min_epochs 40
  --reco_huber_delta 0.10
  --num_hypotheses 6
  --winner_mode reco
  --joint_epochs 12
  --joint_lr 1.2e-4
  --joint_patience 6
  --selector_epochs 45
  --selector_lr 2e-3
  --selector_patience 8
  --selector_rank_weight 0.20
  --selector_rank_margin 0.25
  --use_coord_residual_param
  --enable_scale_sensitive_weighting
  --scale_pt_power 0.85
  --scale_weight_cap 6.0
  --angle_pt_power 1.0
  --loss_w_ar 1.0
  --loss_w_set 1.0
  --set_loss_mode chamfer
  --set_unmatched_penalty 0.35
  --loss_w_best_set 2.8
  --loss_w_diversity 0.10
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
echo "Model-23 MultiHyp K=6 (winner=reco)"
echo "Run:   ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
