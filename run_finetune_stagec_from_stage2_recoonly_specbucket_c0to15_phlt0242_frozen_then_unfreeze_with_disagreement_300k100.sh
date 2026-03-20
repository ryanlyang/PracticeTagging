#!/usr/bin/env bash
# Stage-C reco-only continuation from saved Stage2 with specialist bucket weighting:
#   Bucket: 0 < n_const_hlt <= 15 and p_hlt >= 0.242 (no jet-pt gate by default)
#   Weights: in-bucket y=0 -> 10, in-bucket y=1 -> 4, out-bucket -> 1
#
# Flow:
# 1) Train reco-only classifier with reconstructor frozen (specialist weighting enabled).
# 2) Unfreeze reconstructor and continue reco-only joint finetuning (same weighting).
# 3) Run Teacher/HLT/RecoOnly disagreement + bucket analysis at TPR=50%
#    for both frozen-selected and final selected checkpoints.
#
# Submit:
#   sbatch run_finetune_stagec_from_stage2_recoonly_specbucket_c0to15_phlt0242_frozen_then_unfreeze_with_disagreement_300k100.sh

#SBATCH -J stgCRSpec
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 5:00:00
#SBATCH -o offline_reconstructor_logs/stagec_recoonly_specbucket/stagec_recoonly_specbucket_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_recoonly_specbucket/stagec_recoonly_specbucket_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/stagec_recoonly_specbucket}"
mkdir -p "${LOG_DIR}"

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_300kJ100C}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_recoonly}"
RUN_NAME="${RUN_NAME:-stagec_recoonly_specbucket_c0to15_phlt0242_frozen_then_unfreeze_300k100_seed0}"

N_TRAIN_JETS="${N_TRAIN_JETS:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-70}"
STAGEC_FREEZE_EPOCHS="${STAGEC_FREEZE_EPOCHS:-20}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-12}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_CLS="${STAGEC_LR_CLS:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

# Specialist bucket rule/weights.
SPEC_COUNT_LOW="${SPEC_COUNT_LOW:-0}"
SPEC_COUNT_HIGH="${SPEC_COUNT_HIGH:-15}"
SPEC_PHLT_THR="${SPEC_PHLT_THR:-0.242}"
SPEC_PT_HLT_MIN="${SPEC_PT_HLT_MIN:-0.0}"
SPEC_W_NEG="${SPEC_W_NEG:-10.0}"
SPEC_W_POS="${SPEC_W_POS:-4.0}"
SPEC_W_OTHER="${SPEC_W_OTHER:-1.0}"

# Disagreement analysis settings.
AN_N_EVAL_JETS="${AN_N_EVAL_JETS:-300000}"
AN_OFFSET_EVAL_JETS="${AN_OFFSET_EVAL_JETS:--1}"
AN_TARGET_TPR="${AN_TARGET_TPR:-0.50}"
AN_NUM_WORKERS="${AN_NUM_WORKERS:-1}"
AN_BUCKET_MIN_COUNT="${AN_BUCKET_MIN_COUNT:-1000}"
AN_BUCKET_MIN_POS="${AN_BUCKET_MIN_POS:-300}"
AN_BUCKET_MIN_NEG="${AN_BUCKET_MIN_NEG:-300}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUT_RUN_DIR="${SAVE_DIR}/${RUN_NAME}"

echo "============================================================"
echo "Stage-C reco-only SPECIALIST frozen->unfreeze + disagreement analysis"
echo "Source run: ${RUN_DIR}"
echo "Output run: ${OUT_RUN_DIR}"
echo "Train setup: n_train=${N_TRAIN_JETS}, offset=${OFFSET_JETS}, max_constits=${MAX_CONSTITS}, seed=${SEED}"
echo "Stage-C: epochs=${STAGEC_EPOCHS}, freeze_epochs=${STAGEC_FREEZE_EPOCHS}, patience=${STAGEC_PATIENCE}, min_epochs=${STAGEC_MIN_EPOCHS}"
echo "LRs/loss: lr_cls=${STAGEC_LR_CLS}, lr_reco=${STAGEC_LR_RECO}, lambda_reco=${LAMBDA_RECO}, lambda_cons=${LAMBDA_CONS}, select=${SELECTION_METRIC}"
echo "Specialist bucket: ${SPEC_COUNT_LOW} < n_const_hlt <= ${SPEC_COUNT_HIGH}, p_hlt >= ${SPEC_PHLT_THR}, jet_pt_hlt >= ${SPEC_PT_HLT_MIN}"
echo "Specialist weights: w_neg=${SPEC_W_NEG}, w_pos=${SPEC_W_POS}, w_other=${SPEC_W_OTHER}"
echo "Corrected view: merge/eff flags ON"
echo "Analysis: n_eval=${AN_N_EVAL_JETS}, target_tpr=${AN_TARGET_TPR}"
echo "============================================================"

python finetune_stagec_from_stage2_recoonly.py \
  --run_dir "${RUN_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --stageC_epochs "${STAGEC_EPOCHS}" \
  --stageC_freeze_reco_epochs "${STAGEC_FREEZE_EPOCHS}" \
  --stageC_patience "${STAGEC_PATIENCE}" \
  --stageC_min_epochs "${STAGEC_MIN_EPOCHS}" \
  --stageC_lr_cls "${STAGEC_LR_CLS}" \
  --stageC_lr_reco "${STAGEC_LR_RECO}" \
  --lambda_reco "${LAMBDA_RECO}" \
  --lambda_cons "${LAMBDA_CONS}" \
  --selection_metric "${SELECTION_METRIC}" \
  --use_corrected_flags \
  --spec_bucket_enable \
  --spec_bucket_count_low "${SPEC_COUNT_LOW}" \
  --spec_bucket_count_high "${SPEC_COUNT_HIGH}" \
  --spec_bucket_p_hlt_threshold "${SPEC_PHLT_THR}" \
  --spec_bucket_jet_pt_hlt_min "${SPEC_PT_HLT_MIN}" \
  --spec_bucket_w_neg "${SPEC_W_NEG}" \
  --spec_bucket_w_pos "${SPEC_W_POS}" \
  --spec_bucket_w_other "${SPEC_W_OTHER}" \
  --device "${DEVICE}"

echo
echo "Running disagreement analysis for reco-only specialist frozen checkpoint..."
python analyze_teacher_hlt_joint_disagreements.py \
  --run_dir "${OUT_RUN_DIR}" \
  --save_subdir "disagreement_fpr50_recoonly_spec_stagec_frozen" \
  --joint_mode recoonly \
  --joint_recoonly_ckpt_name "recoonly_classifier_stagec_frozen_ckpt.pt" \
  --reco_ckpt_name "offline_reconstructor_recoonly_stagec_frozen_ckpt.pt" \
  --n_eval_jets "${AN_N_EVAL_JETS}" \
  --offset_eval_jets "${AN_OFFSET_EVAL_JETS}" \
  --target_tpr "${AN_TARGET_TPR}" \
  --threshold_source test \
  --num_workers "${AN_NUM_WORKERS}" \
  --device "${DEVICE}" \
  --bucket_min_count "${AN_BUCKET_MIN_COUNT}" \
  --bucket_min_pos "${AN_BUCKET_MIN_POS}" \
  --bucket_min_neg "${AN_BUCKET_MIN_NEG}"

echo
echo "Running disagreement analysis for reco-only specialist selected checkpoint..."
python analyze_teacher_hlt_joint_disagreements.py \
  --run_dir "${OUT_RUN_DIR}" \
  --save_subdir "disagreement_fpr50_recoonly_spec_stagec_selected" \
  --joint_mode recoonly \
  --joint_recoonly_ckpt_name "recoonly_classifier_stagec_selected_ckpt.pt" \
  --reco_ckpt_name "offline_reconstructor_recoonly_stagec_selected_ckpt.pt" \
  --n_eval_jets "${AN_N_EVAL_JETS}" \
  --offset_eval_jets "${AN_OFFSET_EVAL_JETS}" \
  --target_tpr "${AN_TARGET_TPR}" \
  --threshold_source test \
  --num_workers "${AN_NUM_WORKERS}" \
  --device "${DEVICE}" \
  --bucket_min_count "${AN_BUCKET_MIN_COUNT}" \
  --bucket_min_pos "${AN_BUCKET_MIN_POS}" \
  --bucket_min_neg "${AN_BUCKET_MIN_NEG}"

echo
echo "Done."
echo "Reco-only specialist Stage-C metrics: ${OUT_RUN_DIR}/stagec_recoonly_refine_metrics.json"
echo "Frozen disagreement summary: ${OUT_RUN_DIR}/disagreement_fpr50_recoonly_spec_stagec_frozen/disagreement_summary.json"
echo "Selected disagreement summary: ${OUT_RUN_DIR}/disagreement_fpr50_recoonly_spec_stagec_selected/disagreement_summary.json"
