#!/usr/bin/env bash
#SBATCH --job-name=jcFrLg
#SBATCH --partition=tier3
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --output=offline_reconstructor_logs/fresh_fusion_audit_fixedhlt_filename_large_%j.out
#SBATCH --error=offline_reconstructor_logs/fresh_fusion_audit_fixedhlt_filename_large_%j.err

set -euo pipefail

REPO="${REPO:-/home/ryreu/atlas/PracticeTagging}"
cd "${REPO}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-atlas_kd}"

mkdir -p offline_reconstructor_logs

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
ROOT="${ROOT:-checkpoints/jetclass_joint_dualview}"
PREFIX="${PREFIX:-${ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_fixedhlt_filename}"
OUT_DIR="${OUT_DIR:-${ROOT}/fusion_reports/fresh_audit_fixedhlt_filename_large_250k50k500k}"

STACK_TRAIN_JETS="${STACK_TRAIN_JETS:-250000}"
STACK_VAL_JETS="${STACK_VAL_JETS:-50000}"
FINAL_TEST_JETS="${FINAL_TEST_JETS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STACK_N_JOBS="${STACK_N_JOBS:-1}"

CMD=(
  python -u fresh_fusion_audit/fresh_jetclass_fusion_audit.py
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --source_split test
  --stack_train_jets "${STACK_TRAIN_JETS}"
  --stack_val_jets "${STACK_VAL_JETS}"
  --final_test_jets "${FINAL_TEST_JETS}"
  --device cuda
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --stack_features logits_probs
  --stack_cv 5
  --stack_max_iter 2000
  --stack_n_jobs "${STACK_N_JOBS}"
  --stack_Cs 0.03 0.1 0.3 1.0 3.0 10.0
  --weight_random_samples 2500
  --run_controls
  --source "hlt_baseline:baseline_hlt:${PREFIX}_core01_base"
  --source "offline_teacher:offline_teacher:${PREFIX}_core01_base"
  --source "m2_base:stage2:${PREFIX}_core01_base"
  --source "m2_consstrong:stage2:${PREFIX}_core02_consstrong"
  --source "m2_budgetlite:stage2:${PREFIX}_core03_budgetlite"
  --source "m2_genlow:stage2:${PREFIX}_core04_genlow"
  --source "m2_genhigh:stage2:${PREFIX}_core05_genhigh"
  --source "m2_splitstrong:stage2:${PREFIX}_core06_splitstrong"
  --source "m2_splitlight:stage2:${PREFIX}_core07_splitlight"
  --source "m2_physstrong:stage2:${PREFIX}_core08_physstrong"
  --source "m2_offdropmid:stage2:${PREFIX}_core09_offdropmid"
  --source "m2_offdrophigh:stage2:${PREFIX}_core10_offdrophigh"
  --source "m2_topk60ish:stage2:${PREFIX}_core11_topk60ish"
  --source "m2_antioverlap:stage2:${PREFIX}_core12_antioverlap"
  --stack_group "m2_only:m2_base,m2_consstrong,m2_budgetlite,m2_genlow,m2_genhigh,m2_splitstrong,m2_splitlight,m2_physstrong,m2_offdropmid,m2_offdrophigh,m2_topk60ish,m2_antioverlap"
  --stack_group "hlt_plus_m2:hlt_baseline,m2_base,m2_consstrong,m2_budgetlite,m2_genlow,m2_genhigh,m2_splitstrong,m2_splitlight,m2_physstrong,m2_offdropmid,m2_offdrophigh,m2_topk60ish,m2_antioverlap"
)

echo "============================================================"
echo "Fresh fusion audit: fixed-HLT filename LARGE"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Split: stack_train=${STACK_TRAIN_JETS}, stack_val=${STACK_VAL_JETS}, final_test=${FINAL_TEST_JETS}"
echo "Out: ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
