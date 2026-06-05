#!/usr/bin/env bash
#SBATCH --job-name=jcTD5F
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_targetdrop5_stacked_fusion_250k_m2hybrid_fixedhlt_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_targetdrop5_stacked_fusion_250k_m2hybrid_fixedhlt_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_joint_dualview}"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/fusion_reports/targetdrop5_250k50k250k_m2hybrid_fixedhlt_stacked_acc}"
SCRIPT="${SCRIPT:-$(pwd)/analyze_jetclass_four_model_stacked_fusion_m2hlt.py}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
WEIGHT_SEARCH_MODE="${WEIGHT_SEARCH_MODE:-grid}"
MAX_WEIGHT_CANDIDATES="${MAX_WEIGHT_CANDIDATES:-200000}"
WEIGHT_RANDOM_SAMPLES="${WEIGHT_RANDOM_SAMPLES:-2500}"
WEIGHT_RANDOM_SEED="${WEIGHT_RANDOM_SEED:-52}"
STACK_FEATURES="${STACK_FEATURES:-logits_probs}"
STACK_CV="${STACK_CV:-5}"
STACK_MAX_ITER="${STACK_MAX_ITER:-2000}"
STACK_N_JOBS="${STACK_N_JOBS:-1}"
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0}"

MODEL_01_SPEC="${MODEL_01_SPEC:-tdrop000:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop000}"
MODEL_02_SPEC="${MODEL_02_SPEC:-tdrop015:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop015}"
MODEL_03_SPEC="${MODEL_03_SPEC:-tdrop040:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop040}"
MODEL_04_SPEC="${MODEL_04_SPEC:-tdrop055:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop055}"
MODEL_05_SPEC="${MODEL_05_SPEC:-tdrop070:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop070}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "$(dirname "${OUT_DIR}")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

CMD=(
  python -u "${SCRIPT}"
  --model "${MODEL_01_SPEC}"
  --model "${MODEL_02_SPEC}"
  --model "${MODEL_03_SPEC}"
  --model "${MODEL_04_SPEC}"
  --model "${MODEL_05_SPEC}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --weight_step "${WEIGHT_STEP}"
  --weight_search_mode "${WEIGHT_SEARCH_MODE}"
  --max_weight_candidates "${MAX_WEIGHT_CANDIDATES}"
  --weight_random_samples "${WEIGHT_RANDOM_SAMPLES}"
  --weight_random_seed "${WEIGHT_RANDOM_SEED}"
  --optimize_for "${OPTIMIZE_FOR}"
  --stack_features "${STACK_FEATURES}"
  --stack_cv "${STACK_CV}"
  --stack_max_iter "${STACK_MAX_ITER}"
  --stack_n_jobs "${STACK_N_JOBS}"
  --stack_Cs ${STACK_CS}
)

echo "============================================================"
echo "JetClass TargetDrop5 Stacked Fusion (250k/50k/250k fixed-HLT m2-hybrid)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Data dir: ${DATA_DIR}"
echo "Out dir:  ${OUT_DIR}"
echo "Models:"
echo "  ${MODEL_01_SPEC}"
echo "  ${MODEL_02_SPEC}"
echo "  ${MODEL_03_SPEC}"
echo "  ${MODEL_04_SPEC}"
echo "  ${MODEL_05_SPEC}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"

