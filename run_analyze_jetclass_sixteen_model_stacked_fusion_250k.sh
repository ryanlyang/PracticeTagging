#!/usr/bin/env bash
#SBATCH --job-name=jc16Fuse
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_sixteen_model_stacked_fusion_250k_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_sixteen_model_stacked_fusion_250k_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_joint_dualview}"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/fusion_reports/sixteen_model_250k_stacked_acc}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
STACK_FEATURES="${STACK_FEATURES:-logits_probs}"
STACK_CV="${STACK_CV:-5}"
STACK_MAX_ITER="${STACK_MAX_ITER:-2000}"
STACK_N_JOBS="${STACK_N_JOBS:--1}"
STACK_CS="${STACK_CS:-0.03 0.1 0.3 1.0 3.0 10.0}"

# Exactly 16 model specs, each in form:
#   name:kind:run_dir
# kind in {baseline_hlt, stage2, joint, reco_only_stagea}
MODEL_01_SPEC="${MODEL_01_SPEC:-v1_base:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core01_base}"
MODEL_02_SPEC="${MODEL_02_SPEC:-v1_joint:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core02_joint}"
MODEL_03_SPEC="${MODEL_03_SPEC:-v1_genlow:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core03_genlow}"
MODEL_04_SPEC="${MODEL_04_SPEC:-v1_genhigh:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core04_genhigh}"
MODEL_05_SPEC="${MODEL_05_SPEC:-v1_splitstrong:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core05_splitstrong}"
MODEL_06_SPEC="${MODEL_06_SPEC:-v1_splitlight:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core06_splitlight}"
MODEL_07_SPEC="${MODEL_07_SPEC:-path_prejoint:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core07_path_prejoint}"
MODEL_08_SPEC="${MODEL_08_SPEC:-path_joint:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core08_path_joint}"
MODEL_09_SPEC="${MODEL_09_SPEC:-path_sparsegen_low:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core09_path_sparsegen_low}"
MODEL_10_SPEC="${MODEL_10_SPEC:-path_sparsegen_high:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core10_path_sparsegen_high}"
MODEL_11_SPEC="${MODEL_11_SPEC:-autoteach_base:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core11_autoteach_base}"
MODEL_12_SPEC="${MODEL_12_SPEC:-autoteach_teacherdom:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core12_autoteach_teacherdom}"
MODEL_13_SPEC="${MODEL_13_SPEC:-autoteach_setheavy:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core13_autoteach_setheavy}"
MODEL_14_SPEC="${MODEL_14_SPEC:-path_lcons003_explicit:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
MODEL_15_SPEC="${MODEL_15_SPEC:-autoteacher_explicit:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_stagea_teacherlogit_autoteacher_recoonlydual}"
MODEL_16_SPEC="${MODEL_16_SPEC:-offlineteacher_explicit:stage2:${SAVE_ROOT}/jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_offlineteacher_stagea_recoonlydual}"

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
  --model "${MODEL_01_SPEC}"
  --model "${MODEL_02_SPEC}"
  --model "${MODEL_03_SPEC}"
  --model "${MODEL_04_SPEC}"
  --model "${MODEL_05_SPEC}"
  --model "${MODEL_06_SPEC}"
  --model "${MODEL_07_SPEC}"
  --model "${MODEL_08_SPEC}"
  --model "${MODEL_09_SPEC}"
  --model "${MODEL_10_SPEC}"
  --model "${MODEL_11_SPEC}"
  --model "${MODEL_12_SPEC}"
  --model "${MODEL_13_SPEC}"
  --model "${MODEL_14_SPEC}"
  --model "${MODEL_15_SPEC}"
  --model "${MODEL_16_SPEC}"
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
echo "JetClass Sixteen-Model Stacked Fusion (250k setup)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Data dir:  ${DATA_DIR}"
echo "Out dir:   ${OUT_DIR}"
echo "Objective: ${OPTIMIZE_FOR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"
