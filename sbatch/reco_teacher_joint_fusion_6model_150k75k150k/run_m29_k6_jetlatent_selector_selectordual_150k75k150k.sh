#!/usr/bin/env bash
#SBATCH --job-name=m29k6d
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m29_k6_jetlatent_selector_selectordual_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m29_k6_jetlatent_selector_selectordual_%j.err

set -euo pipefail
mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model29_k6_jetlatent_selector_selectordual_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model29_k6_jetlatent_selector_selectordual}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

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

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_jetlatent_set2set_m29_k6_selector.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets 375000 --n_train_split 150000 --n_val_split 75000 --n_test_split 150000 --offset_jets 0 --max_constits 100
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" --selection_metric auc
  --stageC_progressive_unfreeze --stageC_unfreeze_phase1_epochs 3 --stageC_unfreeze_phase2_epochs 7 --stageC_unfreeze_last_n_encoder_layers 2
  --stageC_lambda_param_anchor 0.02 --stageC_lambda_output_anchor 0.02 --stageC_anchor_decay 0.97
  --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06
  --loss_unselected_penalty 0.0 --loss_gen_local_radius 0.0
  --stageC_lambda_delta 0.05 --stageC_delta_tau 0.05 --stageC_delta_lambda_fp 3.0 --stageC_delta_warmup_epochs 8
  --added_target_scale 0.90 --save_fusion_scores --disable_final_kd
  --m29_num_hypotheses 6 --m29_winner_mode hybrid --m29_winner_alpha 1.0 --m29_winner_beta 0.6
  --m29_loss_w_best_set 2.5 --m29_loss_w_diversity 0.08
  --m29_selector_epochs 30 --m29_selector_lr 2e-3 --m29_selector_patience 8
  --m29_selector_rank_weight 0.20 --m29_selector_rank_margin 0.25
  --m29_stagec_mode selector_dual
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-29 K=6 jetlatent selector routing (StageC selector_dual)"
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}"
