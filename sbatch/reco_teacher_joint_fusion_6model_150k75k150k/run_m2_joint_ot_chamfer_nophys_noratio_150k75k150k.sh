#!/usr/bin/env bash
#SBATCH --job-name=m2joc0
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_ot_chamfer_nophys_noratio_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m2_joint_ot_chamfer_nophys_noratio_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model2_joint_ot_chamfer_nophys_noratio_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_ot_chamfer_nophys_noratio}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-150000}"
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
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
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
  --stageB_lambda_rank 0.0
  --stageB_lambda_cons 0.0
  --stageC_lr_dual 1e-5
  --stageC_lr_reco 5e-6
  --lambda_reco 0.4
  --lambda_cons 0.06
  --stageC_lambda_delta 0.05
  --stageC_delta_tau 0.05
  --stageC_delta_lambda_fp 3.0
  --stageC_delta_warmup_epochs 8
  --added_target_scale 0.90
  --loss_set_mode combo
  --loss_set_combo_w_chamfer 1.0
  --loss_set_combo_w_sinkhorn 0.0
  --loss_set_combo_w_ot 1.0
  --loss_set_combo_w_hungarian 0.0
  --loss_set_sinkhorn_eps 0.08
  --loss_set_sinkhorn_iters 25
  --loss_set_ot_eps 0.02
  --loss_set_ot_iters 50
  --loss_w_phys 0.0
  --loss_w_pt_ratio 0.0
  --loss_w_e_ratio 0.0
  --loss_w_local 0.05
  --save_fusion_scores
  --disable_final_kd
  --device "${DEVICE}"
)

echo "============================================================"
echo "Model-2 Joint (Combo: OT + Chamfer, L_phys=L_pt_ratio=L_e_ratio=0)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "Split: train=${N_TRAIN_SPLIT}, val=${N_VAL_SPLIT}, test=${N_TEST_SPLIT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
