#!/usr/bin/env bash
#SBATCH --job-name=m34k12g
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --time=7-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m34_k12_globalcand_multiview3_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m34_k12_globalcand_multiview3_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model34_k12_globalcand_multiview3_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model34_globalcand_multiview}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-100000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-80}"

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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m34_globalcand_multiview.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --save_fusion_scores
  --conf_weight_floor 1.0
  --prior_lr 5e-5
  --degrader_lr 5e-5
  --prior_loss_w_count 1.0
  --degrader_loss_w_count 1.0
  --proposer_k0_train 96
  --proposer_top_m_train 16
  --proposer_k_wrong 0
  --proposer_lambda_margin 0.0
  --k0_infer 160
  --top_m_infer 24
  --infer_refine_steps 3
  --infer_refine_lr 0.035
  --search_target_k 12
  --search_batch_size 40
  --search_chunk_k0 100
  --search_shortlist_m 20
  --search_max_rounds 20
  --search_keep_per_round 10
  --search_max_pool_size 120
  --search_eps_total 0.20
  --search_eps_count 0.25
  --search_w_chamfer 1.0
  --search_w_count 0.30
  --search_w_pt 0.12
  --search_w_mass 0.06
  --post_accept_refine_steps 6
  --post_accept_refine_lr 0.03
  --post_accept_refine_max_step_norm 0.20
  --post_accept_refine_anchor_lambda 0.10
  --post_accept_refine_accept_margin 0.00
  --post_accept_refine_batch_size 16
  --selector_epochs 45
  --selector_lr 2e-4
  --selector_patience 10
  --selector_neg_per_class 4
  --selector_score_alpha 1.40
  --mv_n_select 3
  --dual_epochs 80
  --dual_lr 1.2e-4
  --dual_patience 14
)

echo "============================================================"
echo "Model-34 K=12 Global Candidate + MultiView3 (PostRefine ON)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
