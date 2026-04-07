#!/usr/bin/env bash
#SBATCH --job-name=m2orhard
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/reco_oracle_route_gate_6model_150k75k150k/m2_orhard_posthoc_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_oracle_route_gate_6model_150k75k150k/m2_orhard_posthoc_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_oracle_route_gate_6model_150k75k150k

RUN_SUBPATH_DEFAULT="reco_teacher_joint_fusion_6model_150k75k150k/model2_oracle_route_gate_moe/model2_oracle_route_gate_moe_150k75k150k_seed0"
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

SCORES_NPZ="${SCORES_NPZ:-}"
OBJECTIVE="${OBJECTIVE:-fpr50}"
DIRECTIONS="${DIRECTIONS:-both}"
NUM_THRESHOLDS="${NUM_THRESHOLDS:-2001}"
TOPK_SWEEP_ROWS="${TOPK_SWEEP_ROWS:-200}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/router_gate_posthoc_hardroute}"

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
if [[ -z "${SCORES_NPZ}" ]]; then
  SCORES_NPZ="${RUN_DIR}/router_gate_scores_val_test.npz"
fi
if [[ ! -f "${SCORES_NPZ}" ]]; then
  echo "ERROR: Scores NPZ not found: ${SCORES_NPZ}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  python analyze_m2_oracle_route_gate_posthoc_hardroute.py
  --run_dir "${RUN_DIR}"
  --scores_npz "${SCORES_NPZ}"
  --objective "${OBJECTIVE}"
  --directions "${DIRECTIONS}"
  --num_thresholds "${NUM_THRESHOLDS}"
  --topk_sweep_rows "${TOPK_SWEEP_ROWS}"
  --out_dir "${OUT_DIR}"
  --report_json "${OUT_DIR}/hardroute_posthoc_report.json"
)

echo "============================================================"
echo "M2 Oracle-Route Posthoc Hard-Route Analysis"
echo "Run dir:  ${RUN_DIR}"
echo "Scores:   ${SCORES_NPZ}"
echo "Objective:${OBJECTIVE}"
echo "Directions:${DIRECTIONS}"
echo "Threshold grid:${NUM_THRESHOLDS}"
echo "Out dir:  ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${OUT_DIR}/hardroute_posthoc_report.json"
echo "  ${OUT_DIR}/hardroute_threshold_sweep_val.csv"
