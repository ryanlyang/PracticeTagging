#!/usr/bin/env bash
#SBATCH --job-name=m2oroute
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=offline_reconstructor_logs/reco_oracle_route_gate_6model_150k75k150k/m2_oracleroute_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_oracle_route_gate_6model_150k75k150k/m2_oracleroute_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_oracle_route_gate_6model_150k75k150k

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

SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_oracle_route_gate_moe}"
RUN_NAME="${RUN_NAME:-model2_oracle_route_gate_moe_150k75k150k_seed0}"
TRAIN_PATH="${TRAIN_PATH:-./data}"

SEED="${SEED:-0}"
HLT_SEED="${HLT_SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-512}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

PHASEA_EPOCHS="${PHASEA_EPOCHS:-8}"
PHASEB_EPOCHS="${PHASEB_EPOCHS:-8}"
PHASEA_LR_GATE="${PHASEA_LR_GATE:-1e-3}"
PHASEB_LR_GATE="${PHASEB_LR_GATE:-5e-4}"
PHASEB_LR_RECO="${PHASEB_LR_RECO:-5e-6}"
PHASEB_UNFREEZE_LAST_N="${PHASEB_UNFREEZE_LAST_N:-2}"

LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
LAMBDA_ROUTE="${LAMBDA_ROUTE:-0.7}"
LAMBDA_RECO_ANCHOR="${LAMBDA_RECO_ANCHOR:-2e-4}"
# Requested off for first run.
LAMBDA_GATE_BALANCE="${LAMBDA_GATE_BALANCE:-0.0}"
GATE_TARGET_USAGE="${GATE_TARGET_USAGE:--1.0}"

GATE_HIDDEN="${GATE_HIDDEN:-256}"
GATE_DROPOUT="${GATE_DROPOUT:-0.10}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

COST_ALPHA_NEG="${COST_ALPHA_NEG:-4.0}"
COST_TAU="${COST_TAU:-0.02}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
CORRUPTIONS="${CORRUPTIONS:-pt_noise:0.04,eta_phi_jitter:0.03,dropout:0.07,merge:0.12,global_scale:0.04}"

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

CMD=(
  python train_m2_oracle_route_gate_moe.py
  --run_dir "${RUN_DIR}"
  --train_path "${TRAIN_PATH}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --hlt_seed "${HLT_SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --selection_metric "${SELECTION_METRIC}"
  --phaseA_epochs "${PHASEA_EPOCHS}"
  --phaseB_epochs "${PHASEB_EPOCHS}"
  --phaseA_lr_gate "${PHASEA_LR_GATE}"
  --phaseB_lr_gate "${PHASEB_LR_GATE}"
  --phaseB_lr_reco "${PHASEB_LR_RECO}"
  --phaseB_unfreeze_last_n_encoder_layers "${PHASEB_UNFREEZE_LAST_N}"
  --lambda_cls "${LAMBDA_CLS}"
  --lambda_route "${LAMBDA_ROUTE}"
  --lambda_reco_anchor "${LAMBDA_RECO_ANCHOR}"
  --lambda_gate_balance "${LAMBDA_GATE_BALANCE}"
  --gate_target_usage "${GATE_TARGET_USAGE}"
  --gate_hidden "${GATE_HIDDEN}"
  --gate_dropout "${GATE_DROPOUT}"
  --cost_alpha_neg "${COST_ALPHA_NEG}"
  --cost_tau "${COST_TAU}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --corruptions "${CORRUPTIONS}"
  --save_fusion_scores
)

echo "============================================================"
echo "M2 Oracle-Route Gate MoE (HLT vs Joint)"
echo "Run dir: ${RUN_DIR}"
echo "Save dir: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train/val/test = ${N_TRAIN_SPLIT}/${N_VAL_SPLIT}/${N_TEST_SPLIT}"
echo "PhaseA/PhaseB epochs: ${PHASEA_EPOCHS}/${PHASEB_EPOCHS}"
echo "Corruptions: ${CORRUPTIONS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

