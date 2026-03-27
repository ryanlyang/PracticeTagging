#!/usr/bin/env bash
#SBATCH --job-name=m14_5view
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m14_fiveview_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m14_fiveview_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"

M2_RUN_DIR="${M2_RUN_DIR:-${BASE_DIR}/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M3_RUN_DIR="${M3_RUN_DIR:-${BASE_DIR}/model3_recoteacher_s09/model3_recoteacher_s09_150k75k150k_seed0}"
M4_RUN_DIR="${M4_RUN_DIR:-${BASE_DIR}/model4_recoteacher_s01_corrected/model4_recoteacher_s01_corrected_150k75k150k_seed0}"
M6_RUN_DIR="${M6_RUN_DIR:-${BASE_DIR}/model6_concat_stagea_corrected/model6_concat_stagea_corrected_150k75k150k_seed0}"

# Explicitly pin m2 to PRE-JOINT checkpoint.
M2_RECO_CKPT="${M2_RECO_CKPT:-offline_reconstructor_stage2.pt}"

RUN_NAME="${RUN_NAME:-model14_fiveview_m2m3m4m6_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-${BASE_DIR}/model14_fiveview_m2m3m4m6}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SELECT_METRIC="${SELECT_METRIC:-auc}"

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

mkdir -p "${SAVE_DIR}"

CMD=(
  python train_fiveview_m2m3m4m6_frozen_then_joint.py
  --m2_run_dir "${M2_RUN_DIR}"
  --m3_run_dir "${M3_RUN_DIR}"
  --m4_run_dir "${M4_RUN_DIR}"
  --m6_run_dir "${M6_RUN_DIR}"
  --m2_reco_ckpt "${M2_RECO_CKPT}"
  --m3_reco_ckpt offline_reconstructor_stageA.pt
  --m4_reco_ckpt offline_reconstructor_stageA.pt
  --m6_reco_ckpt offline_reconstructor_stageA.pt
  --m2_baseline_ckpt baseline.pt
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --corrected_weight_floor 0.03
  --reco_eval_batch_size 256
  --target_tpr 0.50
  --frozen_epochs 40
  --frozen_patience 10
  --frozen_batch_size 256
  --frozen_lr 3e-4
  --frozen_weight_decay 1e-4
  --frozen_warmup_epochs 5
  --frozen_lambda_rank 0.0
  --frozen_rank_tau 0.00
  --joint_epochs 12
  --joint_patience 6
  --joint_batch_size 128
  --joint_lr_tagger 1e-4
  --joint_lr_reco 2e-6
  --joint_weight_decay 1e-4
  --joint_warmup_epochs 3
  --joint_lambda_rank 0.0
  --joint_rank_tau 0.00
  --select_metric "${SELECT_METRIC}"
)

echo "============================================================"
echo "Model-14 Five-view (HLT + m2/m3/m4/m6): frozen then joint"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Selection metric: ${SELECT_METRIC}"
echo "m2 checkpoint (pre-joint): ${M2_RUN_DIR}/${M2_RECO_CKPT}"
echo "m3 checkpoint: ${M3_RUN_DIR}/offline_reconstructor_stageA.pt"
echo "m4 checkpoint: ${M4_RUN_DIR}/offline_reconstructor_stageA.pt"
echo "m6 checkpoint: ${M6_RUN_DIR}/offline_reconstructor_stageA.pt"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
