#!/usr/bin/env bash
#SBATCH --job-name=offrecoJointJR
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:30:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_joint_jetreg_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_joint_jetreg_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-joint_100k_80c_jetreg}"
N_TRAIN_JETS="${N_TRAIN_JETS:-100000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint}"
ENABLE_JET_REGRESSOR="${ENABLE_JET_REGRESSOR:-1}"

JET_REG_ARGS=()
if [ "${ENABLE_JET_REGRESSOR}" = "1" ]; then
  JET_REG_ARGS+=(--enable_jet_regressor)
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "=================================================="
echo "Offline Reconstructor + DualView Joint Training"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Start Time: $(date)"
echo "=================================================="

echo
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo
echo "Running command:"
echo "python offline_reconstructor_joint_dualview.py --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --device cuda ${JET_REG_ARGS[*]}"

echo
python offline_reconstructor_joint_dualview.py \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --device cuda \
  "${JET_REG_ARGS[@]}"

rc=$?
echo
if [ "$rc" -eq 0 ]; then
  echo "=========================================="
  echo "Run completed successfully"
  echo "Results saved to: ${SAVE_DIR}/${RUN_NAME}"
  echo "End Time: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Run failed with exit code: $rc"
  echo "End Time: $(date)"
  echo "=========================================="
fi
exit "$rc"

