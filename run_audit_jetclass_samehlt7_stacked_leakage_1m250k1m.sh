#!/usr/bin/env bash
#SBATCH --job-name=jc7Audit
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_samehlt7_stacked_leakage_audit_1m250k1m_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_samehlt7_stacked_leakage_audit_1m250k1m_%j.err

set -euo pipefail

REPORT_JSON="${REPORT_JSON:-checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/report.json}"
SCORES_NPZ="${SCORES_NPZ:-checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/fusion_scores.npz}"
HLT5_REPORT_JSON="${HLT5_REPORT_JSON:-checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/hlt5_1m250k1m_fixedhlt_stacked_acc/report.json}"
OUT_DIR="${OUT_DIR:-checkpoints/jetclass_joint_dualview/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/leakage_audit}"
DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SCRIPT="${SCRIPT:-$(pwd)/audit_jetclass_samehlt7_stacked_leakage.py}"
DEVICE="${DEVICE:-cuda}"
STACK_N_JOBS="${STACK_N_JOBS:-1}"
HOLDOUT_REPEATS="${HOLDOUT_REPEATS:-3}"
HASH_JETS_PER_SPLIT="${HASH_JETS_PER_SPLIT:-20000}"
INPUT_CONTROL_JETS="${INPUT_CONTROL_JETS:-20000}"
RUN_INPUT_CONTROLS="${RUN_INPUT_CONTROLS:-1}"

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
  --report_json "${REPORT_JSON}"
  --scores_npz "${SCORES_NPZ}"
  --hlt5_report_json "${HLT5_REPORT_JSON}"
  --out_dir "${OUT_DIR}"
  --data_dir "${DATA_DIR}"
  --device "${DEVICE}"
  --stack_n_jobs "${STACK_N_JOBS}"
  --holdout_repeats "${HOLDOUT_REPEATS}"
  --hash_jets_per_split "${HASH_JETS_PER_SPLIT}"
  --input_control_jets "${INPUT_CONTROL_JETS}"
)

if [[ "${RUN_INPUT_CONTROLS}" == "0" ]]; then
  CMD+=(--skip_input_controls)
fi

echo "============================================================"
echo "JetClass Same-HLT 7+HLT Stacked Leakage Audit"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Partition: debug"
echo "Runtime: 1-00:00:00"
echo "Report: ${REPORT_JSON}"
echo "Scores: ${SCORES_NPZ}"
echo "Out:    ${OUT_DIR}"
echo "Input controls: ${RUN_INPUT_CONTROLS} (jets=${INPUT_CONTROL_JETS})"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
