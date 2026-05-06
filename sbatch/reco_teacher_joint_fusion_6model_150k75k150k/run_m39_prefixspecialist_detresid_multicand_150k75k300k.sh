#!/usr/bin/env bash
#SBATCH --job-name=m39sp
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=5-12:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefixspecialist_detresid_multicand_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefixspecialist_detresid_multicand_%j.err

set -euo pipefail

RUN_NAME="${RUN_NAME:-model39_prefixspecialist_detresid_multicand_150k75k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"

N_TRAIN_JETS="${N_TRAIN_JETS:-370000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-50000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-20000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-80}"

SEED_CANDIDATE_K="${SEED_CANDIDATE_K:-1}"
SEED_KEEP_M="${SEED_KEEP_M:-1}"
SEED_MAX_PREFIX="${SEED_MAX_PREFIX:-12}"
TRAIN_SPECIALIST_PREFIX="${TRAIN_SPECIALIST_PREFIX:--1}"
CANDIDATE_GEN_BATCH="${CANDIDATE_GEN_BATCH:-64}"

RECO_SET_LOSS_MODE="${RECO_SET_LOSS_MODE:-chamfer}"
RECO_LR="${RECO_LR:-1.8e-4}"
RECO_BATCH_SIZE="${RECO_BATCH_SIZE:-80}"
RECO_LOSS_W_EOS="${RECO_LOSS_W_EOS:-0.20}"
RECO_LOSS_W_COUNT="${RECO_LOSS_W_COUNT:-0.20}"
RECO_LOSS_W_JETPT="${RECO_LOSS_W_JETPT:-0.08}"
RECO_LOSS_W_4VEC="${RECO_LOSS_W_4VEC:-0.03}"
RECO_LOSS_W_CONF_RANK="${RECO_LOSS_W_CONF_RANK:-0.20}"
RECO_LOSS_W_CONF_PREFIX="${RECO_LOSS_W_CONF_PREFIX:-0.12}"
RECO_CONF_MARGIN="${RECO_CONF_MARGIN:-0.06}"
RECO_CONF_PREFIX_TAU="${RECO_CONF_PREFIX_TAU:-16.0}"

CARRY_EPOCHS="${CARRY_EPOCHS:-48}"
CARRY_PATIENCE="${CARRY_PATIENCE:-10}"
CARRY_LR="${CARRY_LR:-2e-4}"
CARRY_TARGET_MODE="${CARRY_TARGET_MODE:-fixed_k}"
CARRY_TARGET_K="${CARRY_TARGET_K:--1}"
CARRY_TARGET_THRESH_GATE="${CARRY_TARGET_THRESH_GATE:-0}"
CARRY_LR_DECAY_START="${CARRY_LR_DECAY_START:-20}"
CARRY_LR_DECAY_GAMMA="${CARRY_LR_DECAY_GAMMA:-0.96}"
CARRY_MIN_LR_RATIO="${CARRY_MIN_LR_RATIO:-0.35}"
CODEBOOK_PATH="${CODEBOOK_PATH:-}"
CODEBOOK_LABEL="${CODEBOOK_LABEL:-}"
STEP1_QUANTIZE_TEACHER_OFFLINE="${STEP1_QUANTIZE_TEACHER_OFFLINE:-auto}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -n "${PROJECT_ROOT:-}" && -d "${PROJECT_ROOT}" ]]; then
  REPO_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefixspecialist_detresid_multicand.py" ]]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefixspecialist_detresid_multicand.py" ]]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
else
  REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
fi
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${SAVE_DIR}" ]]; then
  SAVE_DIR="${REPO_ROOT}/checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model39_prefixspecialist_detresid_multicand"
fi
mkdir -p "${REPO_ROOT}/offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k"
mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefixspecialist_detresid_multicand.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --save_fusion_scores
  --carry_epochs "${CARRY_EPOCHS}"
  --carry_patience "${CARRY_PATIENCE}"
  --carry_lr "${CARRY_LR}"
  --carry_target_mode "${CARRY_TARGET_MODE}"
  --carry_target_k "${CARRY_TARGET_K}"
  --carry_target_thresh_gate "${CARRY_TARGET_THRESH_GATE}"
  --carry_dist_thresh 0.22
  --carry_lr_decay_start_epoch "${CARRY_LR_DECAY_START}"
  --carry_lr_decay_gamma "${CARRY_LR_DECAY_GAMMA}"
  --carry_min_lr_ratio "${CARRY_MIN_LR_RATIO}"
  --reco_epochs 175
  --reco_patience 30
  --reco_min_epochs 45
  --reco_lr "${RECO_LR}"
  --reco_batch_size "${RECO_BATCH_SIZE}"
  --reco_set_loss_mode "${RECO_SET_LOSS_MODE}"
  --reco_loss_w_eos "${RECO_LOSS_W_EOS}"
  --reco_loss_w_count "${RECO_LOSS_W_COUNT}"
  --reco_loss_w_jetpt "${RECO_LOSS_W_JETPT}"
  --reco_loss_w_4vec "${RECO_LOSS_W_4VEC}"
  --reco_loss_w_conf_rank "${RECO_LOSS_W_CONF_RANK}"
  --reco_loss_w_conf_prefix "${RECO_LOSS_W_CONF_PREFIX}"
  --reco_conf_margin "${RECO_CONF_MARGIN}"
  --reco_conf_prefix_tau "${RECO_CONF_PREFIX_TAU}"
  --reco_physics_warmup_epochs 12
  --reco_phase1_end_epoch 15
  --reco_phase2_end_epoch 75
  --reco_phase3_end_epoch 127
  --reco_phase2_alpha_fr_end 0.70
  --reco_phase3_alpha_fr_end 0.95
  --reco_phase4_alpha_fr 0.95
  --reco_phase2_ss_end 0.60
  --reco_phase3_ss_end 0.90
  --reco_phase4_ss 0.90
  --reco_phase2_free_run_every_n 2
  --reco_phase3_free_run_every_n 1
  --reco_phase4_free_run_every_n 1
  --reco_phase_lr_decay 0.80
  --seed_candidate_k "${SEED_CANDIDATE_K}"
  --seed_keep_m "${SEED_KEEP_M}"
  --seed_max_prefix "${SEED_MAX_PREFIX}"
  --train_specialist_prefix "${TRAIN_SPECIALIST_PREFIX}"
  --seed_temp 0.35
  --candidate_gen_batch "${CANDIDATE_GEN_BATCH}"
  --search_eps_total 0.60
  --search_eps_count 0.30
  --search_w_chamfer 1.00
  --search_w_count 0.25
  --search_w_pt 0.12
  --search_w_mass 0.08
  --dual_epochs 90
  --dual_lr 1.2e-4
  --dual_patience 16
)

if [[ -n "${CODEBOOK_PATH}" ]]; then
  CMD+=(--codebook_path "${CODEBOOK_PATH}")
fi
if [[ -n "${CODEBOOK_LABEL}" ]]; then
  CMD+=(--codebook_label "${CODEBOOK_LABEL}")
fi
if [[ "${STEP1_QUANTIZE_TEACHER_OFFLINE}" == "auto" ]]; then
  if [[ -n "${CODEBOOK_PATH}" ]]; then
    STEP1_QUANTIZE_TEACHER_OFFLINE="1"
  else
    STEP1_QUANTIZE_TEACHER_OFFLINE="0"
  fi
fi
if [[ "${STEP1_QUANTIZE_TEACHER_OFFLINE}" == "1" ]]; then
  CMD+=(--step1_quantize_teacher_offline)
fi

echo "============================================================"
echo "Model-39 Prefix-Specialist + Deterministic-Residual MultiCandidate"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "Reco batch size: ${RECO_BATCH_SIZE}"
echo "Step1 quantized teacher offline: ${STEP1_QUANTIZE_TEACHER_OFFLINE}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
