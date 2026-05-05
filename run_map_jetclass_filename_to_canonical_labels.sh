#!/usr/bin/env bash
#SBATCH --job-name=jcLblMap
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=offline_reconstructor_logs/map_jetclass_filename_to_canonical_labels_%j.out
#SBATCH --error=offline_reconstructor_logs/map_jetclass_filename_to_canonical_labels_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SPLIT="${SPLIT:-all}"  # all | train | val | test
TRAIN_FILES_PER_CLASS="${TRAIN_FILES_PER_CLASS:-8}"
VAL_FILES_PER_CLASS="${VAL_FILES_PER_CLASS:-1}"
TEST_FILES_PER_CLASS="${TEST_FILES_PER_CLASS:-1}"
SEED="${SEED:-52}"

OUTPUT_ROOT="${OUTPUT_ROOT:-plots/jetclass_label_mapping}"
RUN_TAG="${RUN_TAG:-filename_vs_canonical_${SPLIT}}"
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

SCRIPT="${SCRIPT:-$(pwd)/map_jetclass_filename_to_canonical_labels.py}"

python - <<'PY'
import importlib.util
missing = [m for m in ("awkward", "uproot", "numpy") if importlib.util.find_spec(m) is None]
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
  --train_files_per_class "${TRAIN_FILES_PER_CLASS}"
  --val_files_per_class "${VAL_FILES_PER_CLASS}"
  --test_files_per_class "${TEST_FILES_PER_CLASS}"
  --seed "${SEED}"
)

echo "============================================================"
echo "JetClass Filename->Canonical Label Mapping Job"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Split: ${SPLIT}"
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
echo " - ${ABS_OUT}/filename_to_canonical_label_fraction_table.csv"
echo " - ${ABS_OUT}/filename_to_canonical_label_fraction_long.csv"
echo " - ${ABS_OUT}/run_metadata.json"
echo "============================================================"

