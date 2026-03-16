#!/usr/bin/env bash
# Same setup as run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_50k40_lcons006_lreco04.sh
# with merge/eff corrected flags enabled and confidence channels exposed for all moves
# by forcing corrected_weight_floor=0.0.
#
# Submit:
#   sbatch run_offline_reconstructor_joint_dualview_stage2save_auc_norankc_50k40_lcons006_lreco04_flags_confall.sh

#SBATCH --job-name=offAUC50kC
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_auc_50k40_lc006_lr04_flagsconf_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_auc_50k40_lc006_lr04_flagsconf_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_stage2save_auc_norankc_lcons006_lreco04_flags_confall}"
N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running stage2save AUC/no-rankC config (+merge/eff flags + all-moves confidence):"
echo "python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --selection_metric auc --stageB_lambda_rank 0.0 --stageB_lambda_cons 0.0 --stageC_lr_dual 1e-5 --stageC_lr_reco 5e-6 --lambda_reco 0.4 --lambda_cons 0.06 --use_corrected_flags --corrected_weight_floor 0.0 --disable_final_kd --device cuda"

python offline_reconstructor_joint_dualview_stage2save_auc_norankc.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --selection_metric auc \
  --stageB_lambda_rank 0.0 \
  --stageB_lambda_cons 0.0 \
  --stageC_lr_dual 1e-5 \
  --stageC_lr_reco 5e-6 \
  --lambda_reco 0.4 \
  --lambda_cons 0.06 \
  --use_corrected_flags \
  --corrected_weight_floor 0.0 \
  --disable_final_kd \
  --device cuda
