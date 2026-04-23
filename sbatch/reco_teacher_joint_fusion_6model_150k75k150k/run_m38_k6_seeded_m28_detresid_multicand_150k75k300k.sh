#!/usr/bin/env bash
#SBATCH --job-name=m38k6s
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=6-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m38_k6_seeded_m28_detresid_multicand_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m38_k6_seeded_m28_detresid_multicand_%j.err

set -euo pipefail

RUN_NAME="${RUN_NAME:-model38_k6_seeded_m28_detresid_multicand_150k75k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"

N_TRAIN_JETS="${N_TRAIN_JETS:-525000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-80}"

SEED_CANDIDATE_K="${SEED_CANDIDATE_K:-6}"
SEED_KEEP_M="${SEED_KEEP_M:-3}"
SEED_MAX_PREFIX="${SEED_MAX_PREFIX:-12}"
CANDIDATE_GEN_BATCH="${CANDIDATE_GEN_BATCH:-64}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -n "${PROJECT_ROOT:-}" && -d "${PROJECT_ROOT}" ]]; then
  REPO_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m38_seeded_m28_detresid_multicand.py" ]]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m38_seeded_m28_detresid_multicand.py" ]]; then
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

if [[ -z "${SAVE_DIR}" ]]; then
  SAVE_DIR="${REPO_ROOT}/checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model38_seeded_m28_detresid_multicand"
fi
mkdir -p "${REPO_ROOT}/offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k"
mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m38_seeded_m28_detresid_multicand.py
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
  --carry_epochs 24
  --carry_patience 6
  --carry_lr 2e-4
  --carry_dist_thresh 0.22
  --reco_epochs 140
  --reco_patience 20
  --reco_min_epochs 35
  --reco_lr 2e-4
  --reco_batch_size 96
  --reco_set_loss_mode hungarian
  --seed_candidate_k "${SEED_CANDIDATE_K}"
  --seed_keep_m "${SEED_KEEP_M}"
  --seed_max_prefix "${SEED_MAX_PREFIX}"
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

echo "============================================================"
echo "Model-38 K=6 Seeded-m28 + Deterministic-Residual MultiCandidate"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
