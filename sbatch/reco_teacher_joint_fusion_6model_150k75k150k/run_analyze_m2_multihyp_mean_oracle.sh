#!/usr/bin/env bash
#SBATCH --job-name=m2mhmdiag
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_diag_multihyp_mean_oracle_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_diag_multihyp_mean_oracle_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

DEFAULT_RUN_SUBPATH="reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_fulltrain_prog_unfreeze_multihyp_mean/model2_joint_delta005_fulltrain_prog_unfreeze_multihyp_mean_150k75k150k_seed0"
DEFAULT_RUN_DIR_CHECKPOINTS="checkpoints/${DEFAULT_RUN_SUBPATH}"
DEFAULT_RUN_DIR_DOWNLOAD="download_checkpoints/${DEFAULT_RUN_SUBPATH}"

if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -d "${DEFAULT_RUN_DIR_CHECKPOINTS}" ]]; then
    RUN_DIR="${DEFAULT_RUN_DIR_CHECKPOINTS}"
  elif [[ -d "${DEFAULT_RUN_DIR_DOWNLOAD}" ]]; then
    RUN_DIR="${DEFAULT_RUN_DIR_DOWNLOAD}"
  else
    RUN_DIR="${DEFAULT_RUN_DIR_CHECKPOINTS}"
  fi
fi

OUT_DIR="${OUT_DIR:-${RUN_DIR}/multihyp_oracle}"
SPLIT="${SPLIT:-test}"                    # val | test
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_BATCHES="${MAX_BATCHES:--1}"
DEVICE="${DEVICE:-cuda}"
MULTIHYP_TEMPS="${MULTIHYP_TEMPS:-0.85,1.00}"
MULTIHYP_AGG="${MULTIHYP_AGG:-mean}"      # mean | lse
USE_CORRECTED_FLAGS="${USE_CORRECTED_FLAGS:-0}"  # set 1 only if run used corrected flags

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUT_DIR}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 1
fi

COMMON_ARGS=(
  --run_dir "${RUN_DIR}"
  --split "${SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_batches "${MAX_BATCHES}"
  --device "${DEVICE}"
  --multihyp_temps "${MULTIHYP_TEMPS}"
  --multihyp_agg "${MULTIHYP_AGG}"
)

if [[ "${USE_CORRECTED_FLAGS}" == "1" ]]; then
  COMMON_ARGS+=(--use_corrected_flags)
fi

CMD_STAGE2=(
  python analyze_multihyp_mean_oracle_diagnostic.py
  "${COMMON_ARGS[@]}"
  --stage stage2
  --report_json "${OUT_DIR}/stage2_${SPLIT}_multihyp_mean_oracle.json"
)

CMD_JOINT=(
  python analyze_multihyp_mean_oracle_diagnostic.py
  "${COMMON_ARGS[@]}"
  --stage joint
  --report_json "${OUT_DIR}/joint_${SPLIT}_multihyp_mean_oracle.json"
)

echo "============================================================"
echo "MultiHyp Mean Oracle Diagnostic"
echo "Run dir: ${RUN_DIR}"
echo "Split: ${SPLIT}"
echo "Temps: ${MULTIHYP_TEMPS}"
echo "Aggregation: ${MULTIHYP_AGG}"
echo "Device: ${DEVICE}"
echo "Out dir: ${OUT_DIR}"
echo "============================================================"

echo
echo "--- Stage2 diagnostic ---"
printf ' %q' "${CMD_STAGE2[@]}"
echo
"${CMD_STAGE2[@]}"

echo
echo "--- Joint diagnostic ---"
printf ' %q' "${CMD_JOINT[@]}"
echo
"${CMD_JOINT[@]}"

echo
echo "Done. Reports:"
echo "  ${OUT_DIR}/stage2_${SPLIT}_multihyp_mean_oracle.json"
echo "  ${OUT_DIR}/joint_${SPLIT}_multihyp_mean_oracle.json"

