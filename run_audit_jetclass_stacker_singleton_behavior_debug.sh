#!/usr/bin/env bash
#SBATCH --job-name=jcStackAudit
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_stacker_singleton_behavior_audit_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_stacker_singleton_behavior_audit_%j.err

set -euo pipefail

PRIMARY_SCORES="${PRIMARY_SCORES:-checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/fusion_scores.npz}"
PRIMARY_REPORT="${PRIMARY_REPORT:-checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/report.json}"
HLT5_SCORES="${HLT5_SCORES:-checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/hlt5_1m250k1m_fixedhlt_stacked_acc/fusion_scores.npz}"
HLT5_REPORT="${HLT5_REPORT:-checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/hlt5_1m250k1m_fixedhlt_stacked_acc/report.json}"
LEGACY12_SCORES="${LEGACY12_SCORES:-checkpoints/jetclass_joint_dualview/fusion_reports/twelve_model_1m250k1m_m2hybrid_stacked_acc/fusion_scores.npz}"
LEGACY12_REPORT="${LEGACY12_REPORT:-checkpoints/jetclass_joint_dualview/fusion_reports/twelve_model_1m250k1m_m2hybrid_stacked_acc/report.json}"
OUT_DIR="${OUT_DIR:-checkpoints/jetclass_joint_dualview/fusion_reports/stacker_singleton_behavior_audit}"
SCRIPT="${SCRIPT:-$(pwd)/analyze_jetclass_stacker_singleton_audit.py}"

MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-150000}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-300000}"
CONTROL_ROWS="${CONTROL_ROWS:-80000}"
CV="${CV:-5}"
MAX_ITER="${MAX_ITER:-1200}"
DIAG_MAX_ITER="${DIAG_MAX_ITER:-500}"
N_JOBS="${N_JOBS:-1}"
FEATURE_MODES="${FEATURE_MODES:-logits_probs}"
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0}"
SEED="${SEED:-52}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

CMD=(
  python -u "${SCRIPT}"
  --primary_scores "${PRIMARY_SCORES}"
  --primary_report "${PRIMARY_REPORT}"
  --hlt5_scores "${HLT5_SCORES}"
  --hlt5_report "${HLT5_REPORT}"
  --legacy12_scores "${LEGACY12_SCORES}"
  --legacy12_report "${LEGACY12_REPORT}"
  --out_dir "${OUT_DIR}"
  --max_train_rows "${MAX_TRAIN_ROWS}"
  --max_test_rows "${MAX_TEST_ROWS}"
  --control_rows "${CONTROL_ROWS}"
  --cv "${CV}"
  --max_iter "${MAX_ITER}"
  --diag_max_iter "${DIAG_MAX_ITER}"
  --n_jobs "${N_JOBS}"
  --seed "${SEED}"
  --feature_modes ${FEATURE_MODES}
  --Cs ${STACK_CS}
)

echo "============================================================"
echo "JetClass stacker singleton behavior audit"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Partition: debug"
echo "Runtime: 10:00:00"
echo "Primary scores: ${PRIMARY_SCORES}"
echo "HLT5 scores:    ${HLT5_SCORES}"
echo "Legacy12 scores:${LEGACY12_SCORES}"
echo "Out dir:        ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"
