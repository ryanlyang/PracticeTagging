#!/usr/bin/env bash
#SBATCH --job-name=m2teachvr
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_view_eval_6model_150k75k150k/m2_teach_view_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_view_eval_6model_150k75k150k/m2_teach_view_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_view_eval_6model_150k75k150k

RUN_SUBPATH_DEFAULT="reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_hungarian_nophys_noratio/model2_joint_hungarian_nophys_noratio_150k75k150k_seed0"
RUN_DIR_DEFAULT_CHECKPOINTS="checkpoints/${RUN_SUBPATH_DEFAULT}"
RUN_DIR_DEFAULT_DOWNLOAD="download_checkpoints/${RUN_SUBPATH_DEFAULT}"

if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -d "${RUN_DIR_DEFAULT_CHECKPOINTS}" ]]; then
    RUN_DIR="${RUN_DIR_DEFAULT_CHECKPOINTS}"
  elif [[ -d "${RUN_DIR_DEFAULT_DOWNLOAD}" ]]; then
    RUN_DIR="${RUN_DIR_DEFAULT_DOWNLOAD}"
  else
    RUN_DIR="${RUN_DIR_DEFAULT_CHECKPOINTS}"
  fi
fi

TRAIN_PATH="${TRAIN_PATH:-./data}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
RECO_CKPT="${RECO_CKPT:-}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
CORRECTED_USE_FLAGS="${CORRECTED_USE_FLAGS:-0}"
SAVE_SCORES_NPZ="${SAVE_SCORES_NPZ:-0}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/teacher_hlt_vs_reco_stage2_eval}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/data_setup.json" ]]; then
  echo "ERROR: Missing ${RUN_DIR}/data_setup.json" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/data_splits.npz" ]]; then
  echo "ERROR: Missing ${RUN_DIR}/data_splits.npz" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  python analyze_m2_teacher_hlt_vs_reco.py
  --run_dir "${RUN_DIR}"
  --train_path "${TRAIN_PATH}"
  --max_constits "${MAX_CONSTITS}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --out_dir "${OUT_DIR}"
  --report_json "${OUT_DIR}/teacher_hlt_vs_reco_report.json"
)

if [[ -n "${TEACHER_CKPT}" ]]; then
  CMD+=(--teacher_ckpt "${TEACHER_CKPT}")
fi
if [[ -n "${RECO_CKPT}" ]]; then
  CMD+=(--reco_ckpt "${RECO_CKPT}")
fi
if [[ "${CORRECTED_USE_FLAGS}" == "1" ]]; then
  CMD+=(--corrected_use_flags)
fi
if [[ "${SAVE_SCORES_NPZ}" == "1" ]]; then
  CMD+=(--save_scores_npz)
fi

echo "============================================================"
echo "M2 Teacher-On-HLT vs Teacher-On-Reco(Stage2)"
echo "Run dir:    ${RUN_DIR}"
echo "Out dir:    ${OUT_DIR}"
if [[ -n "${TEACHER_CKPT}" ]]; then
  echo "Teacher ckpt override: ${TEACHER_CKPT}"
fi
if [[ -n "${RECO_CKPT}" ]]; then
  echo "Reco ckpt override:    ${RECO_CKPT}"
fi
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${OUT_DIR}/teacher_hlt_vs_reco_report.json"
if [[ "${SAVE_SCORES_NPZ}" == "1" ]]; then
  echo "  ${OUT_DIR}/teacher_hlt_vs_reco_scores.npz"
fi

