#!/usr/bin/env bash
#SBATCH --job-name=m20res
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m20_dualview_residual_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m20_dualview_residual_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model20_dualview_residual_m2m6_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model20_dualview_residual_m2m6}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

M2_RUN_DIR="${M2_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M6_RUN_DIR="${M6_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model6_concat_stagea_corrected/model6_concat_stagea_corrected_150k75k150k_seed0}"

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
  python train_m20_dualview_residual_m2m6.py
  --m2_run_dir "${M2_RUN_DIR}"
  --m6_run_dir "${M6_RUN_DIR}"
  --m2_reco_ckpt offline_reconstructor_stage2.pt
  --m6_reco_ckpt offline_reconstructor_stageA.pt
  --m2_baseline_ckpt baseline.pt
  --teacher_ckpt teacher.pt
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"

  --reco_eval_batch_size 256
  --corrected_weight_floor 0.03
  --target_tpr 0.50

  --frozen_epochs 45
  --frozen_patience 12
  --frozen_batch_size 256
  --frozen_lr 3e-4
  --frozen_weight_decay 1e-4
  --frozen_warmup_epochs 5

  --joint_epochs 12
  --joint_patience 6
  --joint_batch_size 128
  --joint_lr_model 1e-4
  --joint_lr_reco_a 2e-6
  --joint_lr_reco_b 2e-6
  --joint_weight_decay 1e-4
  --joint_warmup_epochs 3

  --lambda_cls 1.0
  --lambda_kd 0.10
  --lambda_residual 0.05
  --lambda_gate 0.01
  --lambda_anchor_a 0.02
  --lambda_anchor_b 0.02
  --kd_temp 2.5

  --select_metric auc
)

echo "============================================================"
echo "Model-20 Dualview Residual (m6 reco + m2 pre-joint reco)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "m2_run_dir: ${M2_RUN_DIR}"
echo "m6_run_dir: ${M6_RUN_DIR}"
echo "Selection metric: val_auc"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
