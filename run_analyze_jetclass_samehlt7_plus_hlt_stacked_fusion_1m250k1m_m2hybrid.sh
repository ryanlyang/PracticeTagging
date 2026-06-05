#!/usr/bin/env bash
#SBATCH --job-name=jc7HLTF
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_samehlt7_plus_hlt_stacked_fusion_1m250k1m_m2hybrid_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_samehlt7_plus_hlt_stacked_fusion_1m250k1m_m2hybrid_%j.err

set -euo pipefail

# Old 1M m2-hybrid runs whose training used the same default HLT corruption,
# plus the HLT baseline from the same setup/core01 run.

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_joint_dualview}"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/fusion_reports/samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc}"
SCRIPT="${SCRIPT:-$(pwd)/analyze_jetclass_four_model_stacked_fusion.py}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
WEIGHT_SEARCH_MODE="${WEIGHT_SEARCH_MODE:-auto}"
MAX_WEIGHT_CANDIDATES="${MAX_WEIGHT_CANDIDATES:-200000}"
WEIGHT_RANDOM_SAMPLES="${WEIGHT_RANDOM_SAMPLES:-2500}"
WEIGHT_RANDOM_SEED="${WEIGHT_RANDOM_SEED:-52}"
STACK_FEATURES="${STACK_FEATURES:-logits_probs}"
STACK_CV="${STACK_CV:-5}"
STACK_MAX_ITER="${STACK_MAX_ITER:-2000}"
STACK_N_JOBS="${STACK_N_JOBS:-1}"
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0}"

CORE01="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core01_base"
CORE02="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core02_consstrong"
CORE03="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core03_budgetlite"
CORE04="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core04_genlow"
CORE05="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core05_genhigh"
CORE11="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core11_topk60ish"
CORE12="${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core12_antioverlap"

MODEL_BASELINE_SPEC="${MODEL_BASELINE_SPEC:-hlt_baseline:baseline_hlt:${CORE01}}"
MODEL_01_SPEC="${MODEL_01_SPEC:-m2_base:stage2:${CORE01}}"
MODEL_02_SPEC="${MODEL_02_SPEC:-m2_consstrong:stage2:${CORE02}}"
MODEL_03_SPEC="${MODEL_03_SPEC:-m2_budgetlite:stage2:${CORE03}}"
MODEL_04_SPEC="${MODEL_04_SPEC:-m2_genlow:stage2:${CORE04}}"
MODEL_05_SPEC="${MODEL_05_SPEC:-m2_genhigh:stage2:${CORE05}}"
MODEL_06_SPEC="${MODEL_06_SPEC:-m2_topk60ish:stage2:${CORE11}}"
MODEL_07_SPEC="${MODEL_07_SPEC:-m2_antioverlap:stage2:${CORE12}}"

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
  --model "${MODEL_BASELINE_SPEC}"
  --model "${MODEL_01_SPEC}"
  --model "${MODEL_02_SPEC}"
  --model "${MODEL_03_SPEC}"
  --model "${MODEL_04_SPEC}"
  --model "${MODEL_05_SPEC}"
  --model "${MODEL_06_SPEC}"
  --model "${MODEL_07_SPEC}"
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
echo "JetClass Same-HLT 7 + HLT Baseline Stacked Fusion (old 1m/250k/1m m2-hybrid)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Data dir:  ${DATA_DIR}"
echo "Out dir:   ${OUT_DIR}"
echo "Objective: ${OPTIMIZE_FOR}"
echo "Models:"
echo "  ${MODEL_BASELINE_SPEC}"
echo "  ${MODEL_01_SPEC}"
echo "  ${MODEL_02_SPEC}"
echo "  ${MODEL_03_SPEC}"
echo "  ${MODEL_04_SPEC}"
echo "  ${MODEL_05_SPEC}"
echo "  ${MODEL_06_SPEC}"
echo "  ${MODEL_07_SPEC}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"
