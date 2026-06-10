#!/usr/bin/env bash
#SBATCH --job-name=jcPtBins
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_pt_decile_bins_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_pt_decile_bins_%j.err

set -euo pipefail

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d /home/ryreu/atlas/PracticeTagging/data/jetclass_part0 ]]; then
    DATA_DIR="/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
  elif [[ -d /home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0 ]]; then
    DATA_DIR="/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"
  else
    DATA_DIR="data/jetclass_part0"
  fi
fi
OUT_ROOT="${OUT_ROOT:-plots/jetclass_pt_decile_bins}"
RUN_TAG="${RUN_TAG:-100k_${PT_MODE:-jet_pt}_${SAMPLING:-class_balanced}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"

N_JETS="${N_JETS:-100000}"
STEP_SIZE="${STEP_SIZE:-20000}"
SAMPLING="${SAMPLING:-class_balanced}"
PT_MODE="${PT_MODE:-jet_pt}"
SCRIPT="${SCRIPT:-$(pwd)/analyze_jetclass_pt_decile_bins.py}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

python - <<'PY'
import importlib.util
missing = [m for m in ("awkward", "uproot", "numpy", "matplotlib") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "[preflight] Missing modules: "
        + ", ".join(missing)
        + ". Install/use an env with uproot, awkward, numpy, and matplotlib."
    )
PY

CMD=(
  python -u "${SCRIPT}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --n_jets "${N_JETS}"
  --step_size "${STEP_SIZE}"
  --sampling "${SAMPLING}"
  --pt_mode "${PT_MODE}"
)

echo "============================================================"
echo "JetClass pT Decile Bin Analysis"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Partition: debug"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "N jets: ${N_JETS}"
echo "pT mode: ${PT_MODE}"
echo "Sampling: ${SAMPLING}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"
