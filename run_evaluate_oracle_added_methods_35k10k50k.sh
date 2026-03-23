#!/usr/bin/env bash
# Oracle sparse-addition comparison runner (IG / LOTO / Greedy).
#
# Submit:
#   sbatch run_evaluate_oracle_added_methods_35k10k50k.sh

#SBATCH --job-name=orclAddMeth
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/oracle_added_methods_35k10k50k_%j.out
#SBATCH --error=offline_reconstructor_logs/oracle_added_methods_35k10k50k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-oracle_added_methods_35k10k50k}"
TRAIN_PATH="${TRAIN_PATH:-$SLURM_SUBMIT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-checkpoints/oracle_added_methods}"

SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

OFFSET_JETS="${OFFSET_JETS:-0}"
N_TRAIN_JETS="${N_TRAIN_JETS:-95000}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
AUG_MAX_CONSTITS="${AUG_MAX_CONSTITS:-100}"

N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-35000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-10000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-50000}"

METHODS="${METHODS:-ig,loto,greedy}"
K_VALUES="${K_VALUES:-8,12,16,20}"

NOVEL_DR_MATCH="${NOVEL_DR_MATCH:-0.02}"
IG_STEPS="${IG_STEPS:-8}"
LOTO_POOL="${LOTO_POOL:-24}"
GREEDY_POOL="${GREEDY_POOL:-12}"
GREEDY_GAIN_MIN="${GREEDY_GAIN_MIN:-0.0}"

# HLT generation knobs
MERGE_RADIUS="${MERGE_RADIUS:-0.01}"
EFF_PLATEAU_BARREL="${EFF_PLATEAU_BARREL:-0.98}"
EFF_PLATEAU_ENDCAP="${EFF_PLATEAU_ENDCAP:-0.94}"
SMEAR_A="${SMEAR_A:-0.35}"
SMEAR_B="${SMEAR_B:-0.012}"
SMEAR_C="${SMEAR_C:-0.08}"

# Optional training overrides; keep -1 for script defaults.
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
echo "Oracle added-methods run (IG/LOTO/Greedy)"
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
echo "python evaluate_oracle_added_methods.py --train_path ${TRAIN_PATH} --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --seed ${SEED} --device ${DEVICE} --num_workers ${NUM_WORKERS} --offset_jets ${OFFSET_JETS} --n_train_jets ${N_TRAIN_JETS} --max_constits ${MAX_CONSTITS} --aug_max_constits ${AUG_MAX_CONSTITS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --methods ${METHODS} --k_values ${K_VALUES} --novel_dr_match ${NOVEL_DR_MATCH} --ig_steps ${IG_STEPS} --loto_pool ${LOTO_POOL} --greedy_pool ${GREEDY_POOL} --greedy_gain_min ${GREEDY_GAIN_MIN} --merge_radius ${MERGE_RADIUS} --eff_plateau_barrel ${EFF_PLATEAU_BARREL} --eff_plateau_endcap ${EFF_PLATEAU_ENDCAP} --smear_a ${SMEAR_A} --smear_b ${SMEAR_B} --smear_c ${SMEAR_C} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --warmup_epochs ${WARMUP_EPOCHS}"
echo ""

python evaluate_oracle_added_methods.py \
  --train_path "${TRAIN_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --offset_jets "${OFFSET_JETS}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --aug_max_constits "${AUG_MAX_CONSTITS}" \
  --n_train_split "${N_TRAIN_SPLIT}" \
  --n_val_split "${N_VAL_SPLIT}" \
  --n_test_split "${N_TEST_SPLIT}" \
  --methods "${METHODS}" \
  --k_values "${K_VALUES}" \
  --novel_dr_match "${NOVEL_DR_MATCH}" \
  --ig_steps "${IG_STEPS}" \
  --loto_pool "${LOTO_POOL}" \
  --greedy_pool "${GREEDY_POOL}" \
  --greedy_gain_min "${GREEDY_GAIN_MIN}" \
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
