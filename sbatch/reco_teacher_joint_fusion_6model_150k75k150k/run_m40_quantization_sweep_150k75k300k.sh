#!/usr/bin/env bash
#SBATCH --job-name=m40qsw
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m40_quant_sweep_150k75k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m40_quant_sweep_150k75k300k_%j.err

set -euo pipefail

RUN_NAME="${RUN_NAME:-m40_quant_sweep_150k75k300k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/m40_constituent_codebook}"

SEED="${SEED:-0}"
USE_TRAIN_WEIGHTS="${USE_TRAIN_WEIGHTS:-1}"

N_TRAIN_JETS="${N_TRAIN_JETS:-525000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

STRATEGIES="${STRATEGIES:-global,pt_stratified,residual2}"
K_VALUES="${K_VALUES:-128,256,512,1024}"
TOP_N="${TOP_N:-3}"

MAX_FIT_JETS="${MAX_FIT_JETS:-120000}"
MAX_FIT_TOKENS="${MAX_FIT_TOKENS:-2500000}"
PT_N_BANDS="${PT_N_BANDS:-4}"
RESIDUAL_COARSE_K="${RESIDUAL_COARSE_K:--1}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k
mkdir -p "${SAVE_DIR}"

CMD=(
  python m40_run_quantization_sweep.py
  --train_path ./data
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --strategies "${STRATEGIES}"
  --k_values "${K_VALUES}"
  --top_n "${TOP_N}"
  --max_fit_jets "${MAX_FIT_JETS}"
  --max_fit_tokens "${MAX_FIT_TOKENS}"
  --pt_n_bands "${PT_N_BANDS}"
  --residual_coarse_k "${RESIDUAL_COARSE_K}"
  --eval_split val
  --emit_launcher
)

if [[ "${USE_TRAIN_WEIGHTS}" == "1" ]]; then
  CMD+=(--use_train_weights)
fi

echo "============================================================"
echo "Model-40 Constituent Codebook Sweep"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Strategies: ${STRATEGIES}"
echo "K values:   ${K_VALUES}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
