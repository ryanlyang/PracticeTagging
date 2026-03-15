#!/usr/bin/env bash
#SBATCH --job-name=offrecoNPM
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/offline_reco_nopriv_merge_only_%j.out
#SBATCH --error=offline_reconstructor_logs/offline_reco_nopriv_merge_only_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-nopriv_staged_merge_only_50k_hltlike}"
N_TRAIN_JETS="${N_TRAIN_JETS:-50000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
NUM_WORKERS="${NUM_WORKERS:-6}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

echo "=================================================="
echo "Offline Reconstructor (No-Priv, Staged, Merge-Only)"
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
echo "python offline_reconstructor_no_gt_nopriv_staged_merge_only.py --save_dir checkpoints/offline_reconstructor_no_gt_nopriv_merge_only --run_name ${RUN_NAME} --n_train_jets ${N_TRAIN_JETS} --offset_jets ${OFFSET_JETS} --max_constits ${MAX_CONSTITS} --num_workers ${NUM_WORKERS} --device cuda"

echo
python offline_reconstructor_no_gt_nopriv_staged_merge_only.py \
  --save_dir checkpoints/offline_reconstructor_no_gt_nopriv_merge_only \
  --run_name "${RUN_NAME}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --offset_jets "${OFFSET_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --num_workers "${NUM_WORKERS}" \
  --device cuda

rc=$?
echo
if [ "$rc" -eq 0 ]; then
  echo "=========================================="
  echo "Run completed successfully"
  echo "Results saved to: checkpoints/offline_reconstructor_no_gt_nopriv_merge_only/${RUN_NAME}"
  echo "End Time: $(date)"
  echo "=========================================="
else
  echo "=========================================="
  echo "Run failed with exit code: $rc"
  echo "End Time: $(date)"
  echo "=========================================="
fi
exit "$rc"
