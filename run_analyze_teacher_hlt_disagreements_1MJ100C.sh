#!/usr/bin/env bash
# Quick disagreement analysis:
# - loads teacher + HLT baseline from a saved run directory
# - rebuilds exact split/setup
# - exports detailed diagnostics + disagreement jet subsets
#
# Submit:
#   sbatch run_analyze_teacher_hlt_disagreements_1MJ100C.sh

#SBATCH --job-name=discAna1M
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/disagreement_analysis_1MJ100C_%j.out
#SBATCH --error=offline_reconstructor_logs/disagreement_analysis_1MJ100C_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_1MJ100C}"
TARGET_TPR="${TARGET_TPR:-0.50}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-val}"   # val or test
NUM_WORKERS="${NUM_WORKERS:-6}"
DEVICE="${DEVICE:-cuda}"
MAX_EXPORT_PER_SUBSET="${MAX_EXPORT_PER_SUBSET:-50000}"
EXPORT_ALL_SUBSET_JETS="${EXPORT_ALL_SUBSET_JETS:-0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cmd=(
  python analyze_teacher_hlt_disagreements.py
  --run_dir "${RUN_DIR}"
  --target_tpr "${TARGET_TPR}"
  --threshold_source "${THRESHOLD_SOURCE}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --max_export_per_subset "${MAX_EXPORT_PER_SUBSET}"
)

if [[ "${EXPORT_ALL_SUBSET_JETS}" -eq 1 ]]; then
  cmd+=(--export_all_subset_jets)
fi

echo "Running disagreement analysis:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

