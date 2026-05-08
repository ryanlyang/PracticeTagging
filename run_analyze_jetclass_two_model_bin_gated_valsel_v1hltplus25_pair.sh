#!/usr/bin/env bash
#SBATCH --job-name=jc2Fuse
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_two_model_bin_gated_fusion_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_two_model_bin_gated_fusion_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
RUN_A_DIR="${RUN_A_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
RUN_B_DIR="${RUN_B_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25_gentok56}"
OUT_DIR="${OUT_DIR:-checkpoints/jetclass_joint_dualview/fusion_reports/$(basename "${RUN_A_DIR}")__AND__$(basename "${RUN_B_DIR}")__bin_gated_valsel}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WEIGHT_STEP="${WEIGHT_STEP:-0.01}"
N_BINS="${N_BINS:-12}"
MIN_BIN_COUNT="${MIN_BIN_COUNT:-1200}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-sigbg_fpr50}"

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

CMD=(
  python -u analyze_jetclass_two_model_bin_gated_fusion.py
  --run_a_dir "${RUN_A_DIR}"
  --run_b_dir "${RUN_B_DIR}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --weight_step "${WEIGHT_STEP}"
  --n_bins "${N_BINS}"
  --min_bin_count "${MIN_BIN_COUNT}"
  --optimize_for "${OPTIMIZE_FOR}"
)

echo "============================================================"
echo "JetClass Two-Model Bin-Gated Fusion Analysis"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run A dir: ${RUN_A_DIR}"
echo "Run B dir: ${RUN_B_DIR}"
echo "Out dir:   ${OUT_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"

