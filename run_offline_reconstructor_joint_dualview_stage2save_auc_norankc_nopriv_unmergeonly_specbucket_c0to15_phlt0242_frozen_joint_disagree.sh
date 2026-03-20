#!/usr/bin/env bash
# Specialist-bucket run (unmerge-only), with explicit frozen-vs-joint specialist disagreement analysis.
# Bucket priority:
#   0 < n_const_hlt <= 15
#   p_hlt >= 0.242
#
# Uses the same specialist weighting/selection strategy:
#   in bucket, y=0 -> w=10
#   in bucket, y=1 -> w=4
#   out bucket      -> w=1
#   weighted val selection in specialist Stage B/C
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_specbucket_c0to15_phlt0242_frozen_joint_disagree.sh

#SBATCH --job-name=uoSpec015
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_uo_specbucket_c0to15_phlt0242_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_uo_specbucket_c0to15_phlt0242_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_unmergeonly_specbucket_c0to15_phlt0242_rho090_200k50k300k_100c_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
N_TRAIN_JETS="${N_TRAIN_JETS:-550000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-200000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-50000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"
ADDED_TARGET_SCALE="${ADDED_TARGET_SCALE:-0.90}"
DEVICE="${DEVICE:-cuda}"

# Specialist bucket rule / weights requested.
SPEC_COUNT_LOW="${SPEC_COUNT_LOW:-0}"
SPEC_COUNT_HIGH="${SPEC_COUNT_HIGH:-15}"
SPEC_PHLT_THR="${SPEC_PHLT_THR:-0.242}"
SPEC_PT_HLT_MIN="${SPEC_PT_HLT_MIN:-0.0}"
SPEC_W_NEG="${SPEC_W_NEG:-10.0}"
SPEC_W_POS="${SPEC_W_POS:-4.0}"
SPEC_W_OTHER="${SPEC_W_OTHER:-1.0}"

# Analysis settings (TPR=50 target as requested).
AN_N_EVAL_JETS="${AN_N_EVAL_JETS:-${N_TEST_SPLIT}}"
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

OUT_RUN_DIR="${SAVE_DIR}/${RUN_NAME}"

echo "============================================================"
echo "Unmerge-only specialist bucket run + frozen/joint disagreement analysis"
echo "Run dir: ${OUT_RUN_DIR}"
echo "Bucket: ${SPEC_COUNT_LOW} < n_const_hlt <= ${SPEC_COUNT_HIGH}, p_hlt >= ${SPEC_PHLT_THR}, jet_pt_hlt >= ${SPEC_PT_HLT_MIN}"
echo "Weights: w_neg=${SPEC_W_NEG}, w_pos=${SPEC_W_POS}, w_other=${SPEC_W_OTHER}"
echo "Splits: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT} (total load ${N_TRAIN_JETS})"
echo "============================================================"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_specialist_bucket.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --n_train_split "${N_TRAIN_SPLIT}" \
  --n_val_split "${N_VAL_SPLIT}" \
  --n_test_split "${N_TEST_SPLIT}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --stageC_lr_dual 1e-5 \
  --stageC_lr_reco 5e-6 \
  --lambda_reco 0.4 \
  --lambda_cons 0.06 \
  --added_target_scale "${ADDED_TARGET_SCALE}" \
  --spec_bucket_enable \
  --spec_bucket_count_low "${SPEC_COUNT_LOW}" \
  --spec_bucket_count_high "${SPEC_COUNT_HIGH}" \
  --spec_bucket_p_hlt_threshold "${SPEC_PHLT_THR}" \
  --spec_bucket_jet_pt_hlt_min "${SPEC_PT_HLT_MIN}" \
  --spec_bucket_w_neg "${SPEC_W_NEG}" \
  --spec_bucket_w_pos "${SPEC_W_POS}" \
  --spec_bucket_w_other "${SPEC_W_OTHER}" \
  --disable_final_kd \
  --device "${DEVICE}"

echo
echo "Running disagreement analysis for specialist frozen checkpoint (Stage B selected)..."
python analyze_teacher_hlt_joint_disagreements.py \
  --run_dir "${OUT_RUN_DIR}" \
  --save_subdir "disagreement_fpr50_specialist_stage2_frozen" \
  --n_eval_jets "${AN_N_EVAL_JETS}" \
  --offset_eval_jets "${AN_OFFSET_EVAL_JETS}" \
  --target_tpr "${AN_TARGET_TPR}" \
  --threshold_source test \
  --num_workers "${AN_NUM_WORKERS}" \
  --device "${DEVICE}" \
  --bucket_min_count "${AN_BUCKET_MIN_COUNT}" \
  --bucket_min_pos "${AN_BUCKET_MIN_POS}" \
  --bucket_min_neg "${AN_BUCKET_MIN_NEG}" \
  --dual_ckpt_name "dual_joint_specialist_stage2.pt" \
  --reco_ckpt_name "offline_reconstructor_specialist_stage2.pt"

echo
echo "Running disagreement analysis for specialist joint checkpoint (Stage C selected)..."
python analyze_teacher_hlt_joint_disagreements.py \
  --run_dir "${OUT_RUN_DIR}" \
  --save_subdir "disagreement_fpr50_specialist_stagec_joint" \
  --n_eval_jets "${AN_N_EVAL_JETS}" \
  --offset_eval_jets "${AN_OFFSET_EVAL_JETS}" \
  --target_tpr "${AN_TARGET_TPR}" \
  --threshold_source test \
  --num_workers "${AN_NUM_WORKERS}" \
  --device "${DEVICE}" \
  --bucket_min_count "${AN_BUCKET_MIN_COUNT}" \
  --bucket_min_pos "${AN_BUCKET_MIN_POS}" \
  --bucket_min_neg "${AN_BUCKET_MIN_NEG}" \
  --dual_ckpt_name "dual_joint_specialist.pt" \
  --reco_ckpt_name "offline_reconstructor_specialist.pt"

echo
echo "Done."
echo "Frozen specialist summary: ${OUT_RUN_DIR}/disagreement_fpr50_specialist_stage2_frozen/disagreement_summary.json"
echo "Joint specialist summary:  ${OUT_RUN_DIR}/disagreement_fpr50_specialist_stagec_joint/disagreement_summary.json"

