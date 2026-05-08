#!/usr/bin/env bash
#SBATCH --job-name=jcOffFuse
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_offline_fused_student_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_offline_fused_student_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
TARGETS_DIR="${TARGETS_DIR:-checkpoints/jetclass_joint_dualview/fused_targets/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual__AND__jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25_gentok56}"
RUN_REF_DIR="${RUN_REF_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25_gentok56}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_NAME="${RUN_NAME:-jetclass_offline_fused_student_50k25k100k_v1hltplus25_pair}"

SEED="${SEED:-52}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-12}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-4}"
EMBED_DIM="${EMBED_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-6}"
FF_DIM="${FF_DIM:-512}"
DROPOUT="${DROPOUT:-0.1}"
TARGET_KEY="${TARGET_KEY:-probs_fused_bin}"
DISTILL_TEMP="${DISTILL_TEMP:-2.5}"
LAMBDA_KL="${LAMBDA_KL:-1.0}"
LAMBDA_CE="${LAMBDA_CE:-0.08}"
USE_CONF_WEIGHT="${USE_CONF_WEIGHT:-1}"

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

CMD=(
  python -u train_jetclass_offline_fused_student.py
  --targets_dir "${TARGETS_DIR}"
  --run_ref_dir "${RUN_REF_DIR}"
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --embed_dim "${EMBED_DIM}"
  --num_heads "${NUM_HEADS}"
  --num_layers "${NUM_LAYERS}"
  --ff_dim "${FF_DIM}"
  --dropout "${DROPOUT}"
  --target_key "${TARGET_KEY}"
  --distill_temp "${DISTILL_TEMP}"
  --lambda_kl "${LAMBDA_KL}"
  --lambda_ce "${LAMBDA_CE}"
)
if [[ "${USE_CONF_WEIGHT}" == "1" ]]; then
  CMD+=( --use_conf_weight )
fi

echo "============================================================"
echo "Train JetClass Offline Fused Student"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Targets: ${TARGETS_DIR}"
echo "Run ref: ${RUN_REF_DIR}"
echo "Save:    ${SAVE_DIR}/${RUN_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${SAVE_DIR}/${RUN_NAME}"

