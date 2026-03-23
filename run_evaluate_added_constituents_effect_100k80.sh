#!/usr/bin/env bash
# Ablation runner: compare top-tagging on Offline vs HLT vs (Offline+HLT added constituents).
#
# Submit:
#   sbatch run_evaluate_added_constituents_effect_100k80.sh

#SBATCH --job-name=addConstAbl
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=offline_reconstructor_logs/added_constits_ablation_100k80_%j.out
#SBATCH --error=offline_reconstructor_logs/added_constits_ablation_100k80_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-added_constits_100k50k300k_80}"
TRAIN_PATH="${TRAIN_PATH:-$SLURM_SUBMIT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-checkpoints/added_constituents_ablation}"

SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

OFFSET_JETS="${OFFSET_JETS:-0}"
N_TRAIN_JETS="${N_TRAIN_JETS:-450000}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
ADDED_MAX_CONSTITS="${ADDED_MAX_CONSTITS:-160}"

# Exact split counts for this run.
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-100000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-50000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"

# HLT generation knobs (same family as unmerge-only pipeline).
MERGE_RADIUS="${MERGE_RADIUS:-0.01}"
EFF_PLATEAU_BARREL="${EFF_PLATEAU_BARREL:-0.98}"
EFF_PLATEAU_ENDCAP="${EFF_PLATEAU_ENDCAP:-0.94}"
SMEAR_A="${SMEAR_A:-0.35}"
SMEAR_B="${SMEAR_B:-0.012}"
SMEAR_C="${SMEAR_C:-0.08}"

# Optional classifier overrides. Keep defaults by leaving at -1.
BATCH_SIZE="${BATCH_SIZE:--1}"
EPOCHS="${EPOCHS:--1}"
PATIENCE="${PATIENCE:--1}"
LR="${LR:--1}"
WEIGHT_DECAY="${WEIGHT_DECAY:--1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:--1}"

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

echo "=================================================="
echo "Added-constituents ablation run"
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
echo "python evaluate_added_constituents_effect.py --train_path ${TRAIN_PATH} --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --seed ${SEED} --device ${DEVICE} --num_workers ${NUM_WORKERS} --offset_jets ${OFFSET_JETS} --n_train_jets ${N_TRAIN_JETS} --max_constits ${MAX_CONSTITS} --added_max_constits ${ADDED_MAX_CONSTITS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --merge_radius ${MERGE_RADIUS} --eff_plateau_barrel ${EFF_PLATEAU_BARREL} --eff_plateau_endcap ${EFF_PLATEAU_ENDCAP} --smear_a ${SMEAR_A} --smear_b ${SMEAR_B} --smear_c ${SMEAR_C} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --warmup_epochs ${WARMUP_EPOCHS}"
echo ""

python evaluate_added_constituents_effect.py \
  --train_path "${TRAIN_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --offset_jets "${OFFSET_JETS}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --added_max_constits "${ADDED_MAX_CONSTITS}" \
  --n_train_split "${N_TRAIN_SPLIT}" \
  --n_val_split "${N_VAL_SPLIT}" \
  --n_test_split "${N_TEST_SPLIT}" \
  --merge_radius "${MERGE_RADIUS}" \
  --eff_plateau_barrel "${EFF_PLATEAU_BARREL}" \
  --eff_plateau_endcap "${EFF_PLATEAU_ENDCAP}" \
  --smear_a "${SMEAR_A}" \
  --smear_b "${SMEAR_B}" \
  --smear_c "${SMEAR_C}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_epochs "${WARMUP_EPOCHS}"

rc=$?
echo ""
if [ "$rc" -eq 0 ]; then
  echo "Run completed successfully"
  echo "Results: ${SAVE_DIR}/${RUN_NAME}"
else
  echo "Run failed with exit code: $rc"
fi
exit "$rc"
