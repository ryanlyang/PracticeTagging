#!/usr/bin/env bash
#SBATCH --job-name=jcFull17Plt
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/plot_jetclass_full17_features_%j.out
#SBATCH --error=offline_reconstructor_logs/plot_jetclass_full17_features_%j.err

set -euo pipefail

# Data / plotting controls (override via env var when submitting).
DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SPLIT="${SPLIT:-train}"                     # train | val | test
CLASS_ASSIGNMENT="${CLASS_ASSIGNMENT:-canonical_labels}"  # canonical_labels | filename
FEATURE_PREPROCESSING="${FEATURE_PREPROCESSING:-canonical}" # canonical | legacy
N_JETS="${N_JETS:-30000}"                   # fast default
MAX_CONSTITS="${MAX_CONSTITS:-100}"
MAX_CONSTITS_PER_CLASS="${MAX_CONSTITS_PER_CLASS:-150000}"
TRAIN_FILES_PER_CLASS="${TRAIN_FILES_PER_CLASS:-8}"
VAL_FILES_PER_CLASS="${VAL_FILES_PER_CLASS:-1}"
TEST_FILES_PER_CLASS="${TEST_FILES_PER_CLASS:-1}"
SEED="${SEED:-52}"
BINS="${BINS:-80}"
CLIP_Q_LOW="${CLIP_Q_LOW:-0.5}"
CLIP_Q_HIGH="${CLIP_Q_HIGH:-99.5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-plots/jetclass_full17_features}"
RUN_TAG="${RUN_TAG:-${SPLIT}_${N_JETS}j}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

SCRIPT="${SCRIPT:-$(pwd)/plot_jetclass_full17_features.py}"

python - <<'PY'
import importlib.util
mods = ["awkward", "uproot", "matplotlib", "numpy"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "[preflight] Missing modules: "
        + ", ".join(missing)
        + ". Please install in env (atlas_kd)."
    )
PY

CMD=(
  python -u "${SCRIPT}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --class_assignment "${CLASS_ASSIGNMENT}"
  --feature_preprocessing "${FEATURE_PREPROCESSING}"
  --seed "${SEED}"
  --max_constits "${MAX_CONSTITS}"
  --n_jets "${N_JETS}"
  --train_files_per_class "${TRAIN_FILES_PER_CLASS}"
  --val_files_per_class "${VAL_FILES_PER_CLASS}"
  --test_files_per_class "${TEST_FILES_PER_CLASS}"
  --bins "${BINS}"
  --max_constits_per_class "${MAX_CONSTITS_PER_CLASS}"
  --clip_quantile_low "${CLIP_Q_LOW}"
  --clip_quantile_high "${CLIP_Q_HIGH}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( ${EXTRA_ARGS} )
  CMD+=( "${EXTRA_ARR[@]}" )
fi

echo "============================================================"
echo "JetClass Full-17 Feature Plotting Job"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Split: ${SPLIT}"
echo "Class assignment: ${CLASS_ASSIGNMENT}"
echo "Feature preprocessing: ${FEATURE_PREPROCESSING}"
echo "Jets: ${N_JETS}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

ABS_OUT="$(python - <<PY
from pathlib import Path
print(Path("${OUTPUT_DIR}").resolve())
PY
)"

echo "============================================================"
echo "Done."
echo "Download this folder:"
echo "${ABS_OUT}"
echo "Key files:"
echo " - ${ABS_OUT}/full17_feature_distributions_by_class.png"
echo " - ${ABS_OUT}/full17_feature_distributions_by_class.pdf"
echo " - ${ABS_OUT}/full17_feature_stats_by_class.csv"
echo " - ${ABS_OUT}/per_feature_png/   (17 separate feature PNGs)"
echo " - ${ABS_OUT}/run_metadata.json"
echo "============================================================"
