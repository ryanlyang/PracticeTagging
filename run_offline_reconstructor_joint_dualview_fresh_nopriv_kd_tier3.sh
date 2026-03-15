#!/usr/bin/env bash
#SBATCH --job-name=offrecoFNPKD
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_fresh_nopriv_kd_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_fresh_nopriv_kd_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-fresh_nopriv_50k_80c_kd}"
N_TRAIN_JETS="${N_TRAIN_JETS:-50000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

# Performance-oriented defaults (override via env if needed)
STAGEA_EPOCHS="${STAGEA_EPOCHS:-100}"
STAGEA_PATIENCE="${STAGEA_PATIENCE:-20}"
STAGEB_EPOCHS="${STAGEB_EPOCHS:-55}"
STAGEB_PATIENCE="${STAGEB_PATIENCE:-15}"
STAGEB_MIN_EPOCHS="${STAGEB_MIN_EPOCHS:-15}"
STAGEC_EPOCHS="${STAGEC_EPOCHS:-80}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-18}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-30}"
STAGEB_LR_DUAL="${STAGEB_LR_DUAL:-4e-4}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1.5e-4}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-8e-5}"
LAMBDA_RECO="${LAMBDA_RECO:-0.35}"
LAMBDA_RANK="${LAMBDA_RANK:-0.45}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
MAX_SPLIT_CHILDREN="${MAX_SPLIT_CHILDREN:-4}"
W_ALLOC_HARD="${W_ALLOC_HARD:-0.25}"
W_ALLOC_QUOTA="${W_ALLOC_QUOTA:-0.15}"
CURR_MERGE_SHARE_START="${CURR_MERGE_SHARE_START:-0.92}"
CURR_MERGE_SHARE_MID="${CURR_MERGE_SHARE_MID:-0.58}"
CURR_MERGE_SHARE_FINAL="${CURR_MERGE_SHARE_FINAL:-0.50}"

# KD ON (Stage C + final KD)
STAGEC_KD_LAMBDA="${STAGEC_KD_LAMBDA:-0.20}"
STAGEC_KD_TEMP="${STAGEC_KD_TEMP:-7.0}"
STAGED_KD_EPOCHS="${STAGED_KD_EPOCHS:-35}"
STAGED_KD_PATIENCE="${STAGED_KD_PATIENCE:-10}"
STAGED_KD_LR="${STAGED_KD_LR:-3e-4}"

ENABLE_JET_REGRESSOR="${ENABLE_JET_REGRESSOR:-1}"

JET_REG_ARGS=()
if [ "${ENABLE_JET_REGRESSOR}" = "1" ]; then
  JET_REG_ARGS+=(--enable_jet_regressor)
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "=================================================="
echo "Offline Reconstructor Joint (Fresh NoPriv, KD ON)"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Start Time: $(date)"
echo "=================================================="

echo
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo
echo "Running command:"
echo "python offline_reconstructor_joint_dualview_fresh_nopriv.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --stageC_enable_kd --lambda_kd_stageC ${STAGEC_KD_LAMBDA} --stageC_kd_temperature ${STAGEC_KD_TEMP} --stageC_kd_conf_weighted ${JET_REG_ARGS[*]}"

echo
python offline_reconstructor_joint_dualview_fresh_nopriv.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --stageA_epochs "${STAGEA_EPOCHS}" \
  --stageA_patience "${STAGEA_PATIENCE}" \
  --stageB_epochs "${STAGEB_EPOCHS}" \
  --stageB_patience "${STAGEB_PATIENCE}" \
  --stageB_min_epochs "${STAGEB_MIN_EPOCHS}" \
  --stageB_lr_dual "${STAGEB_LR_DUAL}" \
  --stageC_epochs "${STAGEC_EPOCHS}" \
  --stageC_patience "${STAGEC_PATIENCE}" \
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}" \
  --stageC_lr_dual "${STAGEC_LR_DUAL}" \
  --stageC_lr_reco "${STAGEC_LR_RECO}" \
  --lambda_reco "${LAMBDA_RECO}" \
  --lambda_rank "${LAMBDA_RANK}" \
  --lambda_cons "${LAMBDA_CONS}" \
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}" \
  --max_split_children "${MAX_SPLIT_CHILDREN}" \
  --w_alloc_hard "${W_ALLOC_HARD}" \
  --w_alloc_quota "${W_ALLOC_QUOTA}" \
  --curr_merge_share_start "${CURR_MERGE_SHARE_START}" \
  --curr_merge_share_mid "${CURR_MERGE_SHARE_MID}" \
  --curr_merge_share_final "${CURR_MERGE_SHARE_FINAL}" \
  --stageC_enable_kd \
  --lambda_kd_stageC "${STAGEC_KD_LAMBDA}" \
  --stageC_kd_temperature "${STAGEC_KD_TEMP}" \
  --stageC_kd_conf_weighted \
  --stageD_kd_epochs "${STAGED_KD_EPOCHS}" \
  --stageD_kd_patience "${STAGED_KD_PATIENCE}" \
  --stageD_kd_lr "${STAGED_KD_LR}" \
  --device cuda \
  "${JET_REG_ARGS[@]}"

rc=$?
echo
if [ "$rc" -eq 0 ]; then
  echo "=========================================="
  echo "Run completed successfully"
  echo "Results saved to: ${SAVE_DIR}/${RUN_NAME}"
  echo "End Time: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Run failed with exit code: $rc"
  echo "End Time: $(date)"
  echo "=========================================="
fi
exit "$rc"

