#!/usr/bin/env bash
#SBATCH --job-name=jcHLT5
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_hlt5_seed_ensemble_1m250k1m_fixedhlt_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_hlt5_seed_ensemble_1m250k1m_fixedhlt_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_hlt_seed_ensemble}"
RUN_PREFIX="${RUN_PREFIX:-hlt5_1m250k1m_fixedhlt_seed}"
SCRIPT="${SCRIPT:-$(pwd)/train_jetclass_hlt_seed_ensemble.py}"

DATA_SEED="${DATA_SEED:-52}"
TRAIN_SEEDS="${TRAIN_SEEDS:-101 202 303 404 505}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"

N_TRAIN_JETS="${N_TRAIN_JETS:-1000000}"
N_VAL_JETS="${N_VAL_JETS:-250000}"
N_TEST_JETS="${N_TEST_JETS:-1000000}"
MAX_CONSTITS="${MAX_CONSTITS:-128}"
FEATURE_MODE="${FEATURE_MODE:-full}"
FEATURE_PREPROCESSING="${FEATURE_PREPROCESSING:-canonical}"
CLASS_ASSIGNMENT="${CLASS_ASSIGNMENT:-filename}"
TARGET_CLASS="${TARGET_CLASS:-Hbb}"
BACKGROUND_CLASS="${BACKGROUND_CLASS:-QCD}"

HLT_PT_THRESHOLD="${HLT_PT_THRESHOLD:-1.30}"
MERGE_PROB_SCALE="${MERGE_PROB_SCALE:-1.35}"
REASSIGN_SCALE="${REASSIGN_SCALE:-1.00}"
SMEAR_SCALE="${SMEAR_SCALE:-1.00}"
EFF_PLATEAU_BARREL="${EFF_PLATEAU_BARREL:-0.99}"
EFF_PLATEAU_ENDCAP="${EFF_PLATEAU_ENDCAP:-0.97}"
EFF_TURNON_PT="${EFF_TURNON_PT:-1.40}"
EFF_WIDTH_PT="${EFF_WIDTH_PT:-0.20}"

BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-12}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONHASHSEED="${DATA_SEED}"
export PYTHONDONTWRITEBYTECODE=1

CMD=(
  python -u "${SCRIPT}"
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_prefix "${RUN_PREFIX}"
  --data_seed "${DATA_SEED}"
  --train_seeds ${TRAIN_SEEDS}
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --feature_mode "${FEATURE_MODE}"
  --feature_preprocessing "${FEATURE_PREPROCESSING}"
  --class_assignment "${CLASS_ASSIGNMENT}"
  --max_constits "${MAX_CONSTITS}"
  --train_files_per_class 8
  --val_files_per_class 1
  --test_files_per_class 1
  --n_train_jets "${N_TRAIN_JETS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --target_class "${TARGET_CLASS}"
  --background_class "${BACKGROUND_CLASS}"
  --hlt_pt_threshold "${HLT_PT_THRESHOLD}"
  --merge_prob_scale "${MERGE_PROB_SCALE}"
  --reassign_scale "${REASSIGN_SCALE}"
  --smear_scale "${SMEAR_SCALE}"
  --eff_plateau_barrel "${EFF_PLATEAU_BARREL}"
  --eff_plateau_endcap "${EFF_PLATEAU_ENDCAP}"
  --eff_turnon_pt "${EFF_TURNON_PT}"
  --eff_width_pt "${EFF_WIDTH_PT}"
)

echo "============================================================"
echo "JetClass HLT-only 5-seed ensemble control"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Data seed: ${DATA_SEED}"
echo "Train seeds: ${TRAIN_SEEDS}"
echo "Jets: train=${N_TRAIN_JETS} val=${N_VAL_JETS} test=${N_TEST_JETS}"
echo "Save dir: ${SAVE_DIR}"
echo "Run prefix: ${RUN_PREFIX}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

