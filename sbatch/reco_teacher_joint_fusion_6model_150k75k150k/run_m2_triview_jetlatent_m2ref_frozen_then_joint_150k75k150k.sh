#!/usr/bin/env bash
#SBATCH --job-name=m2tri
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_triview_jetlatent_m2ref_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_triview_jetlatent_m2ref_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

JETLATENT_RUN_DIR="${JETLATENT_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta005_fulltrain_prog_unfreeze_jetlatent_set2set/model2_joint_delta005_fulltrain_prog_unfreeze_jetlatent_set2set_150k75k150k_seed0}"
M2_REF_RUN_DIR="${M2_REF_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_stagec_prog_unfreeze_150k75k150k_seed0}"

JETLATENT_RECO_CKPT="${JETLATENT_RECO_CKPT:-offline_reconstructor.pt}"
M2_RECO_CKPT="${M2_RECO_CKPT:-offline_reconstructor.pt}"
M2_BASELINE_CKPT="${M2_BASELINE_CKPT:-baseline.pt}"

RUN_NAME="${RUN_NAME:-model2_triview_jetlatent_m2ref_frozen_then_joint_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_triview_jetlatent_m2ref_frozen_then_joint}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-0.03}"
RECO_EVAL_BATCH_SIZE="${RECO_EVAL_BATCH_SIZE:-256}"
TARGET_TPR="${TARGET_TPR:-0.50}"
SELECT_METRIC="${SELECT_METRIC:-auc}"

FROZEN_EPOCHS="${FROZEN_EPOCHS:-40}"
FROZEN_PATIENCE="${FROZEN_PATIENCE:-10}"
FROZEN_BATCH_SIZE="${FROZEN_BATCH_SIZE:-256}"
FROZEN_LR="${FROZEN_LR:-3e-4}"
FROZEN_WEIGHT_DECAY="${FROZEN_WEIGHT_DECAY:-1e-4}"
FROZEN_WARMUP_EPOCHS="${FROZEN_WARMUP_EPOCHS:-5}"
FROZEN_LAMBDA_RANK="${FROZEN_LAMBDA_RANK:-0.2}"
FROZEN_RANK_TAU="${FROZEN_RANK_TAU:-0.05}"

JOINT_EPOCHS="${JOINT_EPOCHS:-12}"
JOINT_PATIENCE="${JOINT_PATIENCE:-6}"
JOINT_BATCH_SIZE="${JOINT_BATCH_SIZE:-128}"
JOINT_LR_TAGGER="${JOINT_LR_TAGGER:-1e-4}"
JOINT_LR_RECO="${JOINT_LR_RECO:-2e-6}"
JOINT_WEIGHT_DECAY="${JOINT_WEIGHT_DECAY:-1e-4}"
JOINT_WARMUP_EPOCHS="${JOINT_WARMUP_EPOCHS:-3}"
JOINT_LAMBDA_RANK="${JOINT_LAMBDA_RANK:-0.2}"
JOINT_RANK_TAU="${JOINT_RANK_TAU:-0.05}"

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
  python train_m2_triview_jetlatent_m2ref_frozen_then_joint.py
  --jetlatent_run_dir "${JETLATENT_RUN_DIR}"
  --m2_ref_run_dir "${M2_REF_RUN_DIR}"
  --jetlatent_reco_ckpt "${JETLATENT_RECO_CKPT}"
  --m2_reco_ckpt "${M2_RECO_CKPT}"
  --m2_baseline_ckpt "${M2_BASELINE_CKPT}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --reco_eval_batch_size "${RECO_EVAL_BATCH_SIZE}"
  --target_tpr "${TARGET_TPR}"
  --select_metric "${SELECT_METRIC}"
  --frozen_epochs "${FROZEN_EPOCHS}"
  --frozen_patience "${FROZEN_PATIENCE}"
  --frozen_batch_size "${FROZEN_BATCH_SIZE}"
  --frozen_lr "${FROZEN_LR}"
  --frozen_weight_decay "${FROZEN_WEIGHT_DECAY}"
  --frozen_warmup_epochs "${FROZEN_WARMUP_EPOCHS}"
  --frozen_lambda_rank "${FROZEN_LAMBDA_RANK}"
  --frozen_rank_tau "${FROZEN_RANK_TAU}"
  --joint_epochs "${JOINT_EPOCHS}"
  --joint_patience "${JOINT_PATIENCE}"
  --joint_batch_size "${JOINT_BATCH_SIZE}"
  --joint_lr_tagger "${JOINT_LR_TAGGER}"
  --joint_lr_reco "${JOINT_LR_RECO}"
  --joint_weight_decay "${JOINT_WEIGHT_DECAY}"
  --joint_warmup_epochs "${JOINT_WARMUP_EPOCHS}"
  --joint_lambda_rank "${JOINT_LAMBDA_RANK}"
  --joint_rank_tau "${JOINT_RANK_TAU}"
)

echo "============================================================"
echo "Tri-view training: HLT + JetLatentReco + M2RefReco"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "JetLatent run: ${JETLATENT_RUN_DIR}"
echo "M2 reference run: ${M2_REF_RUN_DIR}"
echo "M2 reference reconstructor: ${M2_RECO_CKPT}"
echo "Frozen then joint unfreeze: enabled"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
