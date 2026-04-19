#!/usr/bin/env bash
#SBATCH --job-name=m31k6d
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m31_k6_jetlatent_denoise_ctxselector_allthree_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m31_k6_jetlatent_denoise_ctxselector_allthree_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model31_k6_jetlatent_denoise_ctxselector_allthree_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model31_k6_jetlatent_denoise_ctxselector_allthree}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-100000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"

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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_jetlatent_set2set_m31_denoise_ctxselector.py
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
  --selection_metric auc
  --stageC_progressive_unfreeze
  --stageC_unfreeze_phase1_epochs 3
  --stageC_unfreeze_phase2_epochs 7
  --stageC_unfreeze_last_n_encoder_layers 2
  --stageC_lambda_param_anchor 0.02
  --stageC_lambda_output_anchor 0.02
  --stageC_anchor_decay 0.97
  --stageC_lr_dual 1e-5
  --stageC_lr_reco 5e-6
  --lambda_reco 0.4
  --lambda_cons 0.06
  --loss_unselected_penalty 0.0
  --loss_gen_local_radius 0.0
  --stageC_lambda_delta 0.05
  --stageC_delta_tau 0.05
  --stageC_delta_lambda_fp 3.0
  --stageC_delta_warmup_epochs 8
  --added_target_scale 0.90
  --save_fusion_scores
  --disable_final_kd
  --m29_num_hypotheses 6
  --m29_winner_mode reco
  --m29_winner_alpha 1.0
  --m29_winner_beta 0.6
  --m29_loss_w_best_set 2.5
  --m29_loss_w_diversity 0.08
  --m29_selector_epochs 30
  --m29_selector_lr 2e-3
  --m29_selector_patience 8
  --m29_selector_rank_weight 0.20
  --m29_selector_rank_margin 0.25
  --m29_selector_hidden 128
  --m29_selector_heads 4
  --m29_selector_dropout 0.10
  --m29_selector_ce_weight 1.0
  --m29_selector_kl_weight 0.35
  --m29_selector_soft_temp 0.35
  --m31_denoise_steps 3
  --m31_denoise_init_noise_std 0.03
  --m31_denoise_delta_scale 0.35
  --m31_denoise_step_embed_dim 16
  --m29_stagec_mode all_three
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-31 K=6 jetlatent diffusion-lite denoise + contextual selector routing (StageC all_three)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
