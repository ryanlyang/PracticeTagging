#!/usr/bin/env bash
#SBATCH --job-name=m2marg
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_diag_marginal_signal_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_diag_marginal_signal_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

DEFAULT_RUN_SUBPATH="reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_stagec_prog_unfreeze_150k75k150k_seed0"
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

OUT_DIR="${OUT_DIR:-${RUN_DIR}/marginal_signal}"
SPLIT="${SPLIT:-test}"
TARGET_TPRS="${TARGET_TPRS:-0.30,0.50}"
N_CONF_BINS="${N_CONF_BINS:-10}"
N_DISAGREE_BINS="${N_DISAGREE_BINS:-10}"
GATE_C="${GATE_C:-0.2}"
SAVE_PER_JET_NPZ="${SAVE_PER_JET_NPZ:-1}"

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
if [[ ! -f "${RUN_DIR}/fusion_scores_val_test.npz" ]]; then
  echo "ERROR: Missing fusion score file: ${RUN_DIR}/fusion_scores_val_test.npz" >&2
  exit 1
fi

CMD=(
  python analyze_m2_joint_baseline_marginal_signal.py
  --run_dir "${RUN_DIR}"
  --split "${SPLIT}"
  --target_tprs "${TARGET_TPRS}"
  --n_conf_bins "${N_CONF_BINS}"
  --n_disagree_bins "${N_DISAGREE_BINS}"
  --gate_c "${GATE_C}"
  --out_dir "${OUT_DIR}"
  --report_json "${OUT_DIR}/${SPLIT}_marginal_signal_report.json"
)

if [[ "${SAVE_PER_JET_NPZ}" == "1" ]]; then
  CMD+=(--save_per_jet_npz)
fi

echo "============================================================"
echo "M2 Marginal Signal Diagnostic"
echo "Run dir:      ${RUN_DIR}"
echo "Split:        ${SPLIT}"
echo "Target TPRs:  ${TARGET_TPRS}"
echo "Out dir:      ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${OUT_DIR}/${SPLIT}_marginal_signal_report.json"
echo "  ${OUT_DIR}/${SPLIT}_by_hlt_confidence.csv"
echo "  ${OUT_DIR}/${SPLIT}_by_hlt_joint_disagreement.csv"
if [[ "${SAVE_PER_JET_NPZ}" == "1" ]]; then
  echo "  ${OUT_DIR}/${SPLIT}_marginal_signal_perjet.npz"
fi
