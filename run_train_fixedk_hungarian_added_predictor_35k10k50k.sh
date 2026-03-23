#!/usr/bin/env bash
# Fixed-K Hungarian predictor training/eval runner.
#
# Submit:
#   sbatch run_train_fixedk_hungarian_added_predictor_35k10k50k.sh

#SBATCH --job-name=fixkHung8
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/fixedk_hungarian_35k10k50k_%j.out
#SBATCH --error=offline_reconstructor_logs/fixedk_hungarian_35k10k50k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs

RUN_NAME="${RUN_NAME:-fixedk_hungarian_35k10k50k_k8}"
TRAIN_PATH="${TRAIN_PATH:-$SLURM_SUBMIT_DIR/data}"
SAVE_DIR="${SAVE_DIR:-checkpoints/fixedk_hungarian_predictor}"

SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

OFFSET_JETS="${OFFSET_JETS:-0}"
N_TRAIN_JETS="${N_TRAIN_JETS:-650000}"
MAX_CONSTITS="${MAX_CONSTITS:-80}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-300000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-50000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-300000}"

K_FIXED="${K_FIXED:-8}"
NOVEL_DR_MATCH="${NOVEL_DR_MATCH:-0.02}"
IG_STEPS="${IG_STEPS:-8}"
GREEDY_POOL="${GREEDY_POOL:-12}"
GREEDY_GAIN_MIN="${GREEDY_GAIN_MIN:-0.0}"
AUG_MAX_CONSTITS="${AUG_MAX_CONSTITS:-100}"

# HLT generation knobs
MERGE_RADIUS="${MERGE_RADIUS:-0.01}"
EFF_PLATEAU_BARREL="${EFF_PLATEAU_BARREL:-0.98}"
EFF_PLATEAU_ENDCAP="${EFF_PLATEAU_ENDCAP:-0.94}"
SMEAR_A="${SMEAR_A:-0.35}"
SMEAR_B="${SMEAR_B:-0.012}"
SMEAR_C="${SMEAR_C:-0.08}"

# Predictor model/training knobs
PREDICTOR_EMBED_DIM="${PREDICTOR_EMBED_DIM:-192}"
PREDICTOR_HEADS="${PREDICTOR_HEADS:-8}"
PREDICTOR_LAYERS="${PREDICTOR_LAYERS:-4}"
PREDICTOR_FF_DIM="${PREDICTOR_FF_DIM:-512}"
PREDICTOR_DROPOUT="${PREDICTOR_DROPOUT:-0.1}"

PRED_EPOCHS="${PRED_EPOCHS:-60}"
PRED_PATIENCE="${PRED_PATIENCE:-12}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-256}"
PRED_LR="${PRED_LR:-3e-4}"
PRED_WEIGHT_DECAY="${PRED_WEIGHT_DECAY:-1e-5}"
PRED_WARMUP_EPOCHS="${PRED_WARMUP_EPOCHS:-3}"

LOSS_W_LOGPT="${LOSS_W_LOGPT:-1.0}"
LOSS_W_ETA="${LOSS_W_ETA:-0.6}"
LOSS_W_PHI="${LOSS_W_PHI:-0.6}"
LOSS_W_LOGE="${LOSS_W_LOGE:-0.7}"
LOSS_W_SEP="${LOSS_W_SEP:-0.02}"

ENABLE_JOINT_FINETUNE="${ENABLE_JOINT_FINETUNE:-1}"
JOINT_EPOCHS="${JOINT_EPOCHS:-12}"
JOINT_PATIENCE="${JOINT_PATIENCE:-4}"
JOINT_LR_PRED="${JOINT_LR_PRED:-1e-4}"
JOINT_LR_TAGGER="${JOINT_LR_TAGGER:-2e-5}"

# Optional classifier override for teacher/baseline/added; keep -1 for script defaults.
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
echo "Fixed-K Hungarian predictor run"
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
echo "python train_fixedk_hungarian_added_predictor.py --train_path ${TRAIN_PATH} --save_dir ${SAVE_DIR} --run_name ${RUN_NAME} --seed ${SEED} --device ${DEVICE} --num_workers ${NUM_WORKERS} --offset_jets ${OFFSET_JETS} --n_train_jets ${N_TRAIN_JETS} --max_constits ${MAX_CONSTITS} --n_train_split ${N_TRAIN_SPLIT} --n_val_split ${N_VAL_SPLIT} --n_test_split ${N_TEST_SPLIT} --k_fixed ${K_FIXED} --novel_dr_match ${NOVEL_DR_MATCH} --ig_steps ${IG_STEPS} --greedy_pool ${GREEDY_POOL} --greedy_gain_min ${GREEDY_GAIN_MIN} --aug_max_constits ${AUG_MAX_CONSTITS} --merge_radius ${MERGE_RADIUS} --eff_plateau_barrel ${EFF_PLATEAU_BARREL} --eff_plateau_endcap ${EFF_PLATEAU_ENDCAP} --smear_a ${SMEAR_A} --smear_b ${SMEAR_B} --smear_c ${SMEAR_C} --predictor_embed_dim ${PREDICTOR_EMBED_DIM} --predictor_heads ${PREDICTOR_HEADS} --predictor_layers ${PREDICTOR_LAYERS} --predictor_ff_dim ${PREDICTOR_FF_DIM} --predictor_dropout ${PREDICTOR_DROPOUT} --pred_epochs ${PRED_EPOCHS} --pred_patience ${PRED_PATIENCE} --pred_batch_size ${PRED_BATCH_SIZE} --pred_lr ${PRED_LR} --pred_weight_decay ${PRED_WEIGHT_DECAY} --pred_warmup_epochs ${PRED_WARMUP_EPOCHS} --loss_w_logpt ${LOSS_W_LOGPT} --loss_w_eta ${LOSS_W_ETA} --loss_w_phi ${LOSS_W_PHI} --loss_w_loge ${LOSS_W_LOGE} --loss_w_sep ${LOSS_W_SEP} --joint_epochs ${JOINT_EPOCHS} --joint_patience ${JOINT_PATIENCE} --joint_lr_pred ${JOINT_LR_PRED} --joint_lr_tagger ${JOINT_LR_TAGGER} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} --lr ${LR} --weight_decay ${WEIGHT_DECAY} --warmup_epochs ${WARMUP_EPOCHS}"
echo ""

EXTRA_JOINT_FLAG=""
if [ "${ENABLE_JOINT_FINETUNE}" = "1" ]; then
  EXTRA_JOINT_FLAG="--enable_joint_finetune"
fi

python train_fixedk_hungarian_added_predictor.py \
  --train_path "${TRAIN_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --num_workers "${NUM_WORKERS}" \
  --offset_jets "${OFFSET_JETS}" \
  --n_train_jets "${N_TRAIN_JETS}" \
  --max_constits "${MAX_CONSTITS}" \
  --n_train_split "${N_TRAIN_SPLIT}" \
  --n_val_split "${N_VAL_SPLIT}" \
  --n_test_split "${N_TEST_SPLIT}" \
  --k_fixed "${K_FIXED}" \
  --novel_dr_match "${NOVEL_DR_MATCH}" \
  --ig_steps "${IG_STEPS}" \
  --greedy_pool "${GREEDY_POOL}" \
  --greedy_gain_min "${GREEDY_GAIN_MIN}" \
  --aug_max_constits "${AUG_MAX_CONSTITS}" \
  --merge_radius "${MERGE_RADIUS}" \
  --eff_plateau_barrel "${EFF_PLATEAU_BARREL}" \
  --eff_plateau_endcap "${EFF_PLATEAU_ENDCAP}" \
  --smear_a "${SMEAR_A}" \
  --smear_b "${SMEAR_B}" \
  --smear_c "${SMEAR_C}" \
  --predictor_embed_dim "${PREDICTOR_EMBED_DIM}" \
  --predictor_heads "${PREDICTOR_HEADS}" \
  --predictor_layers "${PREDICTOR_LAYERS}" \
  --predictor_ff_dim "${PREDICTOR_FF_DIM}" \
  --predictor_dropout "${PREDICTOR_DROPOUT}" \
  --pred_epochs "${PRED_EPOCHS}" \
  --pred_patience "${PRED_PATIENCE}" \
  --pred_batch_size "${PRED_BATCH_SIZE}" \
  --pred_lr "${PRED_LR}" \
  --pred_weight_decay "${PRED_WEIGHT_DECAY}" \
  --pred_warmup_epochs "${PRED_WARMUP_EPOCHS}" \
  --loss_w_logpt "${LOSS_W_LOGPT}" \
  --loss_w_eta "${LOSS_W_ETA}" \
  --loss_w_phi "${LOSS_W_PHI}" \
  --loss_w_loge "${LOSS_W_LOGE}" \
  --loss_w_sep "${LOSS_W_SEP}" \
  --joint_epochs "${JOINT_EPOCHS}" \
  --joint_patience "${JOINT_PATIENCE}" \
  --joint_lr_pred "${JOINT_LR_PRED}" \
  --joint_lr_tagger "${JOINT_LR_TAGGER}" \
  ${EXTRA_JOINT_FLAG} \
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
