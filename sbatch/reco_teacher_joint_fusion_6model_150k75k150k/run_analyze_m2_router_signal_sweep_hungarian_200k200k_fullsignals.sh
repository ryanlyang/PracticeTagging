#!/usr/bin/env bash
#SBATCH --job-name=m2router_full
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=offline_reconstructor_logs/reco_router_signal_sweep_6model_200k200k_full/m2_router_signal_full_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_router_signal_sweep_6model_200k200k_full/m2_router_signal_full_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_router_signal_sweep_6model_200k200k_full

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

ROUTER_OFFSET_JETS="${ROUTER_OFFSET_JETS:-375000}"
ROUTER_N_ANALYSIS="${ROUTER_N_ANALYSIS:-200000}"
ROUTER_N_TEST="${ROUTER_N_TEST:-200000}"

REF_OFFSET_JETS="${REF_OFFSET_JETS:-0}"
REF_N_JETS="${REF_N_JETS:-375000}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

SEED="${SEED:-0}"
HLT_SEED="${HLT_SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-384}"
DEVICE="${DEVICE:-cuda}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"

# Richer grid to expose slope/curvature and cross-family consistency.
CORRUPTIONS="${CORRUPTIONS:-pt_noise:0.02,pt_noise:0.04,pt_noise:0.06,eta_phi_jitter:0.015,eta_phi_jitter:0.03,eta_phi_jitter:0.05,dropout:0.03,dropout:0.07,dropout:0.12,merge:0.06,merge:0.12,merge:0.20,global_scale:0.02,global_scale:0.04,global_scale:0.06}"
ROUTER_CAL_FRAC="${ROUTER_CAL_FRAC:-0.2}"
COST_ALPHA_NEG="${COST_ALPHA_NEG:-4.0}"
COST_TAU="${COST_TAU:-0.02}"
TOP_PAIR_SIGNAL_K="${TOP_PAIR_SIGNAL_K:-10}"

OUT_DIR="${OUT_DIR:-${RUN_DIR}/router_signal_sweep_200k200k_fullsignals}"
SAVE_PER_JET_NPZ="${SAVE_PER_JET_NPZ:-0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 1
fi
for f in baseline.pt offline_reconstructor.pt dual_joint.pt data_setup.json hlt_stats.json; do
  if [[ ! -f "${RUN_DIR}/${f}" ]]; then
    echo "ERROR: Missing ${RUN_DIR}/${f}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"

CMD=(
  python analyze_m2_router_signal_sweep.py
  --run_dir "${RUN_DIR}"
  --train_path "${TRAIN_PATH}"
  --router_offset_jets "${ROUTER_OFFSET_JETS}"
  --router_n_analysis "${ROUTER_N_ANALYSIS}"
  --router_n_test "${ROUTER_N_TEST}"
  --ref_offset_jets "${REF_OFFSET_JETS}"
  --ref_n_jets "${REF_N_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --seed "${SEED}"
  --hlt_seed "${HLT_SEED}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --corruptions "${CORRUPTIONS}"
  --router_cal_frac "${ROUTER_CAL_FRAC}"
  --cost_alpha_neg "${COST_ALPHA_NEG}"
  --cost_tau "${COST_TAU}"
  --top_pair_signal_k "${TOP_PAIR_SIGNAL_K}"
  --feature_profile full
  --out_dir "${OUT_DIR}"
  --report_json "${OUT_DIR}/router_signal_sweep_report.json"
)

if [[ "${SAVE_PER_JET_NPZ}" == "1" ]]; then
  CMD+=(--save_per_jet_npz)
fi

echo "============================================================"
echo "M2 Router Signal Sweep (FULL signals, 200k analysis + 200k test)"
echo "Run dir:    ${RUN_DIR}"
echo "Out dir:    ${OUT_DIR}"
echo "Data split: offset=${ROUTER_OFFSET_JETS}, analysis=${ROUTER_N_ANALYSIS}, test=${ROUTER_N_TEST}"
echo "Corruptions: ${CORRUPTIONS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${OUT_DIR}/router_signal_sweep_report.json"
echo "  ${OUT_DIR}/router_leaderboard_test.csv"
echo "  ${OUT_DIR}/router_signal_rank_fit.csv"
if [[ "${SAVE_PER_JET_NPZ}" == "1" ]]; then
  echo "  ${OUT_DIR}/router_signal_perjet.npz"
fi

