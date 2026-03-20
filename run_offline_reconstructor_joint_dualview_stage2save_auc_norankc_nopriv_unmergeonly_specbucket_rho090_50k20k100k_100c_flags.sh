#!/usr/bin/env bash
# Unmerge-only base pipeline + specialist bucket branch.
# - Base branch: standard joint reconstructor + dual-view model.
# - Specialist branch: full-data training with bucket weights:
#   * in bucket, y=0 -> w=10
#   * in bucket, y=1 -> w=4
#   * out bucket -> w=1
# - End-of-run reports two routed variants:
#   1) base=joint dual-view, bucket->specialist
#   2) base=HLT baseline,  bucket->specialist
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_specbucket_rho090_50k20k100k_100c_flags.sh

#SBATCH --job-name=uoSpecB
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_uo_specbucket_200k50k300k100_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_uo_specbucket_200k50k300k100_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_stage2save_auc_norankc_nopriv_unmergeonly_specbucket_c10to20_pt1077216_rho090_200k50k300k_100c_seed0_bucket3}"
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

# Specialist bucket rule / weights
SPEC_COUNT_LOW="${SPEC_COUNT_LOW:-10}"
SPEC_COUNT_HIGH="${SPEC_COUNT_HIGH:-20}"
SPEC_PHLT_THR="${SPEC_PHLT_THR:-0.0}"
SPEC_PT_HLT_MIN="${SPEC_PT_HLT_MIN:-1077216.0}"
SPEC_W_NEG="${SPEC_W_NEG:-10.0}"
SPEC_W_POS="${SPEC_W_POS:-4.0}"
SPEC_W_OTHER="${SPEC_W_OTHER:-1.0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running unmerge-only specialist-bucket pipeline:"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_specialist_bucket.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --seed ${SEED} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06 --added_target_scale ${ADDED_TARGET_SCALE} --spec_bucket_enable --spec_bucket_count_low ${SPEC_COUNT_LOW} --spec_bucket_count_high ${SPEC_COUNT_HIGH} --spec_bucket_p_hlt_threshold ${SPEC_PHLT_THR} --spec_bucket_jet_pt_hlt_min ${SPEC_PT_HLT_MIN} --spec_bucket_w_neg ${SPEC_W_NEG} --spec_bucket_w_pos ${SPEC_W_POS} --spec_bucket_w_other ${SPEC_W_OTHER} --disable_final_kd --device cuda"

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
  --device cuda
