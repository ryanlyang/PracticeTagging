#!/usr/bin/env bash
# Run the Tagging2 pure-unsmear shared-encoder joint setup on SLURM.
# Default is exactly: n_jets=50k, max_particles=40.
#
# Submit:
#   sbatch run_tagging2_pure_unsmear_joint_50j40c.sh

#SBATCH --job-name=unsmJ50kx40
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=offline_reconstructor_logs/unsmear_joint_50k40c_%j.out
#SBATCH --error=offline_reconstructor_logs/unsmear_joint_50k40c_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-unsmear_transformer_sharedencoder_50k_40c}"
DATA_PATH="${DATA_PATH:-$SLURM_SUBMIT_DIR/data/test.h5}"
N_JETS="${N_JETS:-50000}"
MAX_PARTICLES="${MAX_PARTICLES:-40}"
SEED="${SEED:-42}"

# Training settings.
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
PATIENCE="${PATIENCE:-8}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# Keep the best-performing notebook defaults.
KD_ENABLE="${KD_ENABLE:-1}"
KD_TEMPERATURE="${KD_TEMPERATURE:-3.0}"
KD_ALPHA="${KD_ALPHA:-0.5}"
KD_ALPHA_ATTN="${KD_ALPHA_ATTN:-0.0}"
RESMEAR_BASELINES="${RESMEAR_BASELINES:-1}"
RESMEAR_JOINT="${RESMEAR_JOINT:-1}"

# Optional checkpoint-loading behavior.
LOAD_SHARED_BASELINES="${LOAD_SHARED_BASELINES:-0}"
LOAD_JOINT_MODEL="${LOAD_JOINT_MODEL:-0}"

# Execute core pipeline cells only (setup + data + loaders + train/eval + ROC figure).
EXECUTE_UNTIL_CELL="${EXECUTE_UNTIL_CELL:-4}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

BOOL_ARGS=()
if [ "$KD_ENABLE" = "1" ]; then BOOL_ARGS+=(--kd_enable); else BOOL_ARGS+=(--no-kd_enable); fi
if [ "$RESMEAR_BASELINES" = "1" ]; then BOOL_ARGS+=(--resmear_each_epoch_baselines); else BOOL_ARGS+=(--no-resmear_each_epoch_baselines); fi
if [ "$RESMEAR_JOINT" = "1" ]; then BOOL_ARGS+=(--resmear_each_epoch_joint); else BOOL_ARGS+=(--no-resmear_each_epoch_joint); fi
if [ "$LOAD_SHARED_BASELINES" = "1" ]; then BOOL_ARGS+=(--load_shared_baselines); else BOOL_ARGS+=(--no-load_shared_baselines); fi
if [ "$LOAD_JOINT_MODEL" = "1" ]; then BOOL_ARGS+=(--load_joint_model); else BOOL_ARGS+=(--no-load_joint_model); fi

echo "=================================================="
echo "Tagging2 pure-unsmear joint run"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Python: $(which python)"
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

echo ""
echo "Running command:"
echo "python Tagging2/pure_unsmear/joint/run_sharedencoder_notebook.py --data_path ${DATA_PATH} --run_name ${RUN_NAME} --n_jets ${N_JETS} --max_particles ${MAX_PARTICLES} --seed ${SEED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --warmup_epochs ${WARMUP_EPOCHS} --patience ${PATIENCE} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --kd_temperature ${KD_TEMPERATURE} --kd_alpha ${KD_ALPHA} --kd_alpha_attn ${KD_ALPHA_ATTN} --execute_until_cell ${EXECUTE_UNTIL_CELL} ${BOOL_ARGS[*]}"
echo ""

python Tagging2/pure_unsmear/joint/run_sharedencoder_notebook.py \
  --data_path "${DATA_PATH}" \
  --run_name "${RUN_NAME}" \
  --n_jets "${N_JETS}" \
  --max_particles "${MAX_PARTICLES}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --patience "${PATIENCE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --kd_temperature "${KD_TEMPERATURE}" \
  --kd_alpha "${KD_ALPHA}" \
  --kd_alpha_attn "${KD_ALPHA_ATTN}" \
  --execute_until_cell "${EXECUTE_UNTIL_CELL}" \
  "${BOOL_ARGS[@]}"

rc=$?
echo ""
if [ "$rc" -eq 0 ]; then
  echo "Run completed successfully"
  echo "Results: Tagging2/pure_unsmear/joint/runs/${RUN_NAME}"
else
  echo "Run failed with exit code: $rc"
fi
exit "$rc"
