#!/usr/bin/env bash
#SBATCH --job-name=jcG4Resp
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:55:00
#SBATCH --output=offline_reconstructor_logs/jetclass_goal4_jet_response_500k250k1m_hybridops_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_goal4_jet_response_500k250k1m_hybridops_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_joint_dualview}"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/response_reports/goal4_500k250k1m_hybridops_pt_response_fast100k}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-512}"
RESPONSE_N_BINS="${RESPONSE_N_BINS:-8}"
RESPONSE_MIN_COUNT="${RESPONSE_MIN_COUNT:-300}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
SCORE_BIAS_WEIGHT="${SCORE_BIAS_WEIGHT:-1.0}"
SCORE_RESOLUTION_WEIGHT="${SCORE_RESOLUTION_WEIGHT:-1.0}"
MAX_TEST_JETS="${MAX_TEST_JETS:-100000}"

PREFIX="${PREFIX:-${SAVE_ROOT}/jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_fixedhlt}"

MODEL_01_SPEC="${MODEL_01_SPEC:-goal_budgetlite:stage2:${PREFIX}_core03_budgetlite}"
MODEL_02_SPEC="${MODEL_02_SPEC:-goal_splitlight:stage2:${PREFIX}_core07_splitlight}"
MODEL_03_SPEC="${MODEL_03_SPEC:-goal_reassignstrong:stage2:${PREFIX}_core08_reassignstrong}"
MODEL_04_SPEC="${MODEL_04_SPEC:-goal_antioverlap:stage2:${PREFIX}_core12_antioverlap}"

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

CMD=(
  python -u analyze_jetclass_twelve_model_jet_response.py
  --model "${MODEL_01_SPEC}"
  --model "${MODEL_02_SPEC}"
  --model "${MODEL_03_SPEC}"
  --model "${MODEL_04_SPEC}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --response_n_bins "${RESPONSE_N_BINS}"
  --response_min_count "${RESPONSE_MIN_COUNT}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --score_bias_weight "${SCORE_BIAS_WEIGHT}"
  --score_resolution_weight "${SCORE_RESOLUTION_WEIGHT}"
  --plot_all_models
)

if [ "${MAX_TEST_JETS}" != "0" ]; then
  CMD+=(--max_test_jets "${MAX_TEST_JETS}")
fi

echo "============================================================"
echo "JetClass Goal4 pT Response/Resolution (500k/250k/1m hybridops-goal)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Data dir: ${DATA_DIR}"
echo "Out dir:  ${OUT_DIR}"
echo "Max test jets: ${MAX_TEST_JETS}"
echo "Models: goal_budgetlite goal_splitlight goal_reassignstrong goal_antioverlap"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"
