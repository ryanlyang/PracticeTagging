#!/usr/bin/env bash
#SBATCH --job-name=m39s2p6_5m
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefix6_stage2_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefix6_stage2_5m1m1m_%j.err

set -euo pipefail

RUN_NAME="${RUN_NAME:-model39_prefix6_stage2_5m1m1m_seed0}"
SAVE_DIR="${SAVE_DIR:-}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"

N_TRAIN_JETS="${N_TRAIN_JETS:-7000000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-5000000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-1000000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-1000000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"

STAGE1_SAVE_DIR="${STAGE1_SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model39_prefixspecialist_detresid_multicand_5m1m1m}"
STAGE1_RUN_NAMES="${STAGE1_RUN_NAMES:-model39_prefixspecialist_detresid_multicand_5m1m1m_seed0_pfx0,model39_prefixspecialist_detresid_multicand_5m1m1m_seed0_pfx3,model39_prefixspecialist_detresid_multicand_5m1m1m_seed0_pfx6,model39_prefixspecialist_detresid_multicand_5m1m1m_seed0_pfx9,model39_prefixspecialist_detresid_multicand_5m1m1m_seed0_pfx12,model39_prefixspecialist_detresid_multicand_5m1m1m_seed0_pfx15}"
STAGE1_PREFIX_FALLBACKS="${STAGE1_PREFIX_FALLBACKS:-0,3,6,9,12,15}"
STAGE2_KEEP_M="${STAGE2_KEEP_M:-6}"

CANDIDATE_GEN_BATCH="${CANDIDATE_GEN_BATCH:-64}"
DUAL_EPOCHS="${DUAL_EPOCHS:-90}"
DUAL_PATIENCE="${DUAL_PATIENCE:-16}"
DUAL_LR="${DUAL_LR:-1.2e-4}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -n "${PROJECT_ROOT:-}" && -d "${PROJECT_ROOT}" ]]; then
  REPO_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefix6_stage2.py" ]]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
elif [[ -f "${SUBMIT_DIR}/../../offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefix6_stage2.py" ]]; then
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
  SAVE_DIR="${REPO_ROOT}/checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model39_prefix6_stage2_5m1m1m"
fi
mkdir -p "${REPO_ROOT}/offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k"
mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m39_prefix6_stage2.py
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
  --stage1_save_dir "${STAGE1_SAVE_DIR}"
  --stage1_run_names "${STAGE1_RUN_NAMES}"
  --stage1_prefix_fallbacks "${STAGE1_PREFIX_FALLBACKS}"
  --stage2_keep_m "${STAGE2_KEEP_M}"
  --candidate_gen_batch "${CANDIDATE_GEN_BATCH}"
  --search_eps_total 0.60
  --search_eps_count 0.30
  --search_w_chamfer 1.00
  --search_w_count 0.25
  --search_w_pt 0.12
  --search_w_mass 0.08
  --dual_epochs "${DUAL_EPOCHS}"
  --dual_patience "${DUAL_PATIENCE}"
  --dual_lr "${DUAL_LR}"
  --save_fusion_scores
)

echo "============================================================"
echo "Model-39 Prefix6 Stage2 (5M/1M/1M)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Stage1 runs: ${STAGE1_RUN_NAMES}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
