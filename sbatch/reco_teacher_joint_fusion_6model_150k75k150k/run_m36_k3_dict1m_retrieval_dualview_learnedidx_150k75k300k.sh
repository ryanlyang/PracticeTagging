#!/usr/bin/env bash
#SBATCH --job-name=m36drlrn
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m36_k3_dict1m_retrieval_dualview_learnedidx_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m36_k3_dict1m_retrieval_dualview_learnedidx_%j.err

set -euo pipefail

RUN_NAME="${RUN_NAME:-model36_k3_dict1m_retrieval_dualview_learnedidx_150k75k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-1525000}"
N_DICT_SPLIT="${N_DICT_SPLIT:-1000000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-80}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -n "${PROJECT_ROOT:-}" && -d "${PROJECT_ROOT}" ]]; then
  REPO_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m36_dictretrieval_dualview.py" ]]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m36_dictretrieval_dualview.py" ]]; then
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
  SAVE_DIR="${REPO_ROOT}/checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model36_dictretrieval_dualview"
fi
mkdir -p "${REPO_ROOT}/offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k"
mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m36_dictretrieval_dualview.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_dict_split "${N_DICT_SPLIT}"
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
  --retrieval_target_k 3
  --retrieval_per_round 256
  --retrieval_max_rounds 10
  --retrieval_batch_size 256
  --retrieval_eps_total 0.90
  --retrieval_eps_count 0.50
  --retrieval_w_desc 1.00
  --retrieval_w_count 0.25
  --retrieval_w_pt 0.12
  --retrieval_w_mass 0.08
  --retrieval_index_mode learned
  --retrieval_embed_dim 16
  --retrieval_embed_hidden 96
  --retrieval_embed_epochs 8
  --retrieval_embed_batch_size 512
  --retrieval_embed_pool_size 256
  --retrieval_embed_train_anchors 100000
  --retrieval_embed_lr 3e-4
  --retrieval_embed_weight_decay 1e-4
  --retrieval_embed_margin 0.20
  --selector_epochs 45
  --selector_lr 2e-4
  --selector_patience 10
  --selector_neg_per_class 3
  --selector_score_alpha 1.35
  --dual_epochs 80
  --dual_lr 1.2e-4
  --dual_patience 14
)

echo "============================================================"
echo "Model-36 K=3 Dictionary-Retrieval DualView (learned index)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: dict=${N_DICT_SPLIT}, train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
