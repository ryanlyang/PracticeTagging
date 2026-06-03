#!/usr/bin/env bash
#SBATCH --job-name=m2j5w12
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=360G
#SBATCH --time=5-05:00:00
#SBATCH --requeue
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_step1load_joint12_weighted_5m1m1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_delta005_step1load_joint12_weighted_5m1m1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model2_joint_delta005_weighted_5m1m1m_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_weighted_5m1m1m}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TRAIN_PATH="${TRAIN_PATH:-./data/train_quarter.h5}"

N_TRAIN_JETS="${N_TRAIN_JETS:-7000000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-5000000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-1000000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-1000000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

STEP1_LOAD_DIR="${STEP1_LOAD_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/teacher_hlt_only_weighted_5m1m1m/teacher_hlt_only_weighted_5m1m1m_seed0}"
STAGEA_LOAD_RECO_CKPT="${STAGEA_LOAD_RECO_CKPT:-}"
STAGEA_EPOCHS="${STAGEA_EPOCHS:-90}"
STAGEA_PATIENCE="${STAGEA_PATIENCE:-18}"
STAGEB_EPOCHS="${STAGEB_EPOCHS:-45}"
STAGEB_PATIENCE="${STAGEB_PATIENCE:-12}"
STAGEB_MIN_EPOCHS="${STAGEB_MIN_EPOCHS:-12}"
STAGEC_EPOCHS="${STAGEC_EPOCHS:-12}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-12}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-12}"
SKIP_STAGEC_JOINT="${SKIP_STAGEC_JOINT:-0}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${SAVE_DIR}"

if [[ ! -f "${STEP1_LOAD_DIR}/teacher.pt" ]]; then
  echo "ERROR: Missing teacher checkpoint: ${STEP1_LOAD_DIR}/teacher.pt" >&2
  exit 2
fi
if [[ ! -f "${STEP1_LOAD_DIR}/baseline.pt" ]]; then
  echo "ERROR: Missing baseline checkpoint: ${STEP1_LOAD_DIR}/baseline.pt" >&2
  exit 2
fi
if [[ -n "${STAGEA_LOAD_RECO_CKPT}" && ! -f "${STAGEA_LOAD_RECO_CKPT}" ]]; then
  echo "ERROR: Missing Stage-A reconstructor checkpoint: ${STAGEA_LOAD_RECO_CKPT}" >&2
  exit 2
fi

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
  --train_path "${TRAIN_PATH}"
  --use_train_weights
  --force_m5_step1
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --stageA_epochs "${STAGEA_EPOCHS}"
  --stageA_patience "${STAGEA_PATIENCE}"
  --selection_metric auc
  --val_selection_mode unweighted
  --stageB_epochs "${STAGEB_EPOCHS}"
  --stageB_patience "${STAGEB_PATIENCE}"
  --stageB_min_epochs "${STAGEB_MIN_EPOCHS}"
  --stageB_lambda_rank 0.0
  --stageB_lambda_cons 0.0
  --stageC_epochs "${STAGEC_EPOCHS}"
  --stageC_patience "${STAGEC_PATIENCE}"
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}"
  --stageC_lr_dual 1e-5
  --stageC_lr_reco 5e-6
  --lambda_reco 0.4
  --lambda_cons 0.06
  --stageC_lambda_delta 0.05
  --stageC_delta_tau 0.05
  --stageC_delta_lambda_fp 3.0
  --stageC_delta_warmup_epochs 8
  --added_target_scale 0.90
  --save_fusion_scores
  --disable_final_kd
  --step1_load_dir "${STEP1_LOAD_DIR}"
  --device "${DEVICE}"
)

if [[ -n "${STAGEA_LOAD_RECO_CKPT}" ]]; then
  CMD+=(--stageA_load_reco_ckpt "${STAGEA_LOAD_RECO_CKPT}")
fi
if [[ "${SKIP_STAGEC_JOINT}" == "1" ]]; then
  CMD+=(--skip_stageC_joint)
fi

echo "============================================================"
echo "Model-2 Joint weighted (L_delta=0.05), Step-1 loaded, 12-epoch StageC"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Step1 load dir: ${STEP1_LOAD_DIR}"
echo "Train path: ${TRAIN_PATH}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}, n_train_jets=${N_TRAIN_JETS}"
echo "StageA: epochs=${STAGEA_EPOCHS}, patience=${STAGEA_PATIENCE}, load=${STAGEA_LOAD_RECO_CKPT:-<none>}"
echo "StageB: epochs=${STAGEB_EPOCHS}, patience=${STAGEB_PATIENCE}, min_epochs=${STAGEB_MIN_EPOCHS}"
echo "StageC: skip=${SKIP_STAGEC_JOINT}, epochs=${STAGEC_EPOCHS}, patience=${STAGEC_PATIENCE}, min_epochs=${STAGEC_MIN_EPOCHS}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
