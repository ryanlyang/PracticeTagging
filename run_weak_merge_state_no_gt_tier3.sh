#!/usr/bin/env bash
#SBATCH --job-name=weak_mrg
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=weak_merge_state_logs/weak_merge_state_%j.out
#SBATCH --error=weak_merge_state_logs/weak_merge_state_%j.err

set -euo pipefail

mkdir -p weak_merge_state_logs

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

TRAIN_PATH=${TRAIN_PATH:-"./data"}
N_TRAIN_JETS=${N_TRAIN_JETS:-200000}
OFFSET_JETS=${OFFSET_JETS:-0}
MAX_CONSTITS=${MAX_CONSTITS:-80}
MAX_MERGE_COUNT=${MAX_MERGE_COUNT:-10}
SAVE_DIR=${SAVE_DIR:-"checkpoints/weak_merge_state"}
RUN_NAME=${RUN_NAME:-"merge_binary_no_gt"}
NUM_WORKERS=${NUM_WORKERS:-6}
PT_RESOLUTION=${PT_RESOLUTION:-0.02}
ETA_RESOLUTION=${ETA_RESOLUTION:-0.005}
PHI_RESOLUTION=${PHI_RESOLUTION:-0.005}
MERGE_RADIUS=${MERGE_RADIUS:-0.01}
EFFICIENCY_LOSS=${EFFICIENCY_LOSS:-0.0}
PT_THRESHOLD_HLT=${PT_THRESHOLD_HLT:-0.0}

CMD="python weak_merge_state_no_gt.py \
  --train_path ${TRAIN_PATH} \
  --n_train_jets ${N_TRAIN_JETS} \
  --offset_jets ${OFFSET_JETS} \
  --max_constits ${MAX_CONSTITS} \
  --max_merge_count ${MAX_MERGE_COUNT} \
  --save_dir ${SAVE_DIR} \
  --run_name ${RUN_NAME} \
  --num_workers ${NUM_WORKERS} \
  --pt_resolution ${PT_RESOLUTION} \
  --eta_resolution ${ETA_RESOLUTION} \
  --phi_resolution ${PHI_RESOLUTION} \
  --merge_radius ${MERGE_RADIUS} \
  --efficiency_loss ${EFFICIENCY_LOSS} \
  --pt_threshold_hlt ${PT_THRESHOLD_HLT}"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CMD="${CMD} --device cuda"
else
  CMD="${CMD} --device cpu"
fi

echo "Running command:"
echo "${CMD}"
echo ""
eval "${CMD}"
