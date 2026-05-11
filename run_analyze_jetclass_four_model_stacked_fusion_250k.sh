#!/usr/bin/env bash
#SBATCH --job-name=jc4Fuse
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_four_model_stacked_fusion_250k_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_four_model_stacked_fusion_250k_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_joint_dualview}"

# 3 finished setups + 1 baseline source (baseline_hlt checkpoint from this run).
RUN_BASELINE_SRC="${RUN_BASELINE_SRC:-${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_gentok56}"
RUN_PATH_LCONS003="${RUN_PATH_LCONS003:-${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
RUN_V1HLT_PLUS25="${RUN_V1HLT_PLUS25:-${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_gentok56}"
RUN_AUTO_TLOGIT="${RUN_AUTO_TLOGIT:-${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_stagea_teacherlogit_autoteacher_recoonlydual}"

OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/fusion_reports/four_model_250k_stacked_acc}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Fusion tuning.
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"         # acc | auc_macro | sigbg_fpr50 | targetbg_fpr50
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"          # simplex grid step for weighted averages
STACK_FEATURES="${STACK_FEATURES:-logits_probs}"   # logits | probs | logits_probs
STACK_CV="${STACK_CV:-5}"
STACK_MAX_ITER="${STACK_MAX_ITER:-2000}"
STACK_N_JOBS="${STACK_N_JOBS:--1}"
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0}"

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
  python -u analyze_jetclass_four_model_stacked_fusion.py
  --model "baseline:baseline_hlt:${RUN_BASELINE_SRC}"
  --model "path_lcons003:stage2:${RUN_PATH_LCONS003}"
  --model "v1hlt_plus25:stage2:${RUN_V1HLT_PLUS25}"
  --model "auto_teacherlogit:stage2:${RUN_AUTO_TLOGIT}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --weight_step "${WEIGHT_STEP}"
  --optimize_for "${OPTIMIZE_FOR}"
  --stack_features "${STACK_FEATURES}"
  --stack_cv "${STACK_CV}"
  --stack_max_iter "${STACK_MAX_ITER}"
  --stack_n_jobs "${STACK_N_JOBS}"
  --stack_Cs ${STACK_CS}
)

echo "============================================================"
echo "JetClass Four-Model Stacked Fusion (250k setup)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Data dir:       ${DATA_DIR}"
echo "Baseline src:   ${RUN_BASELINE_SRC}"
echo "Model A stage2: ${RUN_PATH_LCONS003}"
echo "Model B stage2: ${RUN_V1HLT_PLUS25}"
echo "Model C stage2: ${RUN_AUTO_TLOGIT}"
echo "Out dir:        ${OUT_DIR}"
echo "Objective:      ${OPTIMIZE_FOR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"

