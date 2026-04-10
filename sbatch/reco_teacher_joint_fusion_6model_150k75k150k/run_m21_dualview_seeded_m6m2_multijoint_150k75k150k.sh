#!/usr/bin/env bash
#SBATCH --job-name=m21seed
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m21_dualview_seeded_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m21_dualview_seeded_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model21_dualview_seeded_m6m2_multijoint_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model21_dualview_seeded_m6m2}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

# Source runs (pre-joint reconstructors)
M2_RUN_DIR="${M2_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_hungarian_nophys_noratio/model2_joint_hungarian_nophys_noratio_150k75k150k_seed0}"
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
  python train_m21_dualview_seeded_m6m2_multijoint.py
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
  --frozen_lambda_rank 0.2
  --frozen_rank_tau 0.05

  --joint_modes both,m6_only,m2_only
  --joint_epochs 25
  --joint_patience 8
  --joint_batch_size 128
  --joint_lr_dual 1e-4
  --joint_lr_reco_a 2e-6
  --joint_lr_reco_b 2e-6
  --joint_weight_decay 1e-4
  --joint_warmup_epochs 3
  --joint_lambda_rank 0.2
  --joint_rank_tau 0.05
  --joint_lambda_anchor_a 0.02
  --joint_lambda_anchor_b 0.02
  --joint_unfreeze_phase1_epochs 3
  --joint_unfreeze_phase2_epochs 7

  --select_metric auc
)

echo "============================================================"
echo "Model-21 Seeded DualView (m6 + m2 Hungarian)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "m6_run_dir: ${M6_RUN_DIR}"
echo "m2_run_dir: ${M2_RUN_DIR}"
echo "Joint modes: both,m6_only,m2_only"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
