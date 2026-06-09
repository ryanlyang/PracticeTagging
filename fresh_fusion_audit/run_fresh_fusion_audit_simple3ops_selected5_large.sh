#!/usr/bin/env bash
#SBATCH --job-name=jcS3FrL
#SBATCH --partition=tier3
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --output=offline_reconstructor_logs/fresh_fusion_audit_simple3ops_selected5_large_%j.out
#SBATCH --error=offline_reconstructor_logs/fresh_fusion_audit_simple3ops_selected5_large_%j.err

set -euo pipefail

REPO="${REPO:-/home/ryreu/atlas/PracticeTagging}"
cd "${REPO}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-atlas_kd}"

mkdir -p offline_reconstructor_logs

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
ROOT="${ROOT:-checkpoints/jetclass_joint_dualview}"
PREFIX="${PREFIX:-${ROOT}/jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_fixedhlt}"
REF_RUN="${REF_RUN:-${PREFIX}_core03_budgetlite}"
OUT_DIR="${OUT_DIR:-${ROOT}/fusion_reports/fresh_audit_simple3ops_selected5_large_250k50k500k}"

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
  --source "hlt_baseline:baseline_hlt:${REF_RUN}"
  --source "offline_teacher:offline_teacher:${REF_RUN}"
  --source "s3_budgetlite:stage2:${PREFIX}_core03_budgetlite"
  --source "s3_splitlight:stage2:${PREFIX}_core07_splitlight"
  --source "s3_reassignstrong:stage2:${PREFIX}_core08_reassignstrong"
  --source "s3_offdrophigh:stage2:${PREFIX}_core10_offdrophigh"
  --source "s3_antioverlap:stage2:${PREFIX}_core12_antioverlap"
  --stack_group "simple3_only:s3_budgetlite,s3_splitlight,s3_reassignstrong,s3_offdrophigh,s3_antioverlap"
  --stack_group "hlt_plus_simple3:hlt_baseline,s3_budgetlite,s3_splitlight,s3_reassignstrong,s3_offdrophigh,s3_antioverlap"
)

echo "============================================================"
echo "Fresh fusion audit: simple3ops selected5 LARGE"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Split: stack_train=${STACK_TRAIN_JETS}, stack_val=${STACK_VAL_JETS}, final_test=${FINAL_TEST_JETS}"
echo "Reference run: ${REF_RUN}"
echo "Out: ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
