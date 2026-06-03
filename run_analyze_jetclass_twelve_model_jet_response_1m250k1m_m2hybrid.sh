#!/usr/bin/env bash
#SBATCH --job-name=jc12Resp
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=offline_reconstructor_logs/jetclass_twelve_model_jet_response_1m250k1m_m2hybrid_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_twelve_model_jet_response_1m250k1m_m2hybrid_%j.err

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/jetclass_joint_dualview}"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/response_reports/twelve_model_1m250k1m_m2hybrid_pt_response}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RESPONSE_N_BINS="${RESPONSE_N_BINS:-8}"
RESPONSE_MIN_COUNT="${RESPONSE_MIN_COUNT:-300}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
SCORE_BIAS_WEIGHT="${SCORE_BIAS_WEIGHT:-1.0}"
SCORE_RESOLUTION_WEIGHT="${SCORE_RESOLUTION_WEIGHT:-1.0}"
MAX_TEST_JETS="${MAX_TEST_JETS:-0}"

MODEL_01_SPEC="${MODEL_01_SPEC:-m2_base:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core01_base}"
MODEL_02_SPEC="${MODEL_02_SPEC:-m2_consstrong:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core02_consstrong}"
MODEL_03_SPEC="${MODEL_03_SPEC:-m2_budgetlite:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core03_budgetlite}"
MODEL_04_SPEC="${MODEL_04_SPEC:-m2_genlow:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core04_genlow}"
MODEL_05_SPEC="${MODEL_05_SPEC:-m2_genhigh:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core05_genhigh}"
MODEL_06_SPEC="${MODEL_06_SPEC:-m2_splitstrong:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core06_splitstrong}"
MODEL_07_SPEC="${MODEL_07_SPEC:-m2_splitlight:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core07_splitlight}"
MODEL_08_SPEC="${MODEL_08_SPEC:-m2_reassignstrong:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core08_reassignstrong}"
MODEL_09_SPEC="${MODEL_09_SPEC:-m2_offdropmid:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core09_offdropmid}"
MODEL_10_SPEC="${MODEL_10_SPEC:-m2_offdrophigh:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core10_offdrophigh}"
MODEL_11_SPEC="${MODEL_11_SPEC:-m2_topk60ish:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core11_topk60ish}"
MODEL_12_SPEC="${MODEL_12_SPEC:-m2_antioverlap:stage2:${SAVE_ROOT}/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core12_antioverlap}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p offline_reconstructor_logs
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

CMD=(
  python -u analyze_jetclass_twelve_model_jet_response.py
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
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --response_n_bins "${RESPONSE_N_BINS}"
  --response_min_count "${RESPONSE_MIN_COUNT}"
  --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
  --score_bias_weight "${SCORE_BIAS_WEIGHT}"
  --score_resolution_weight "${SCORE_RESOLUTION_WEIGHT}"
)

if [ "${MAX_TEST_JETS}" != "0" ]; then
  CMD+=(--max_test_jets "${MAX_TEST_JETS}")
fi

echo "============================================================"
echo "JetClass Twelve-Model pT Response/Resolution (1m/250k/1m m2-hybrid)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Data dir: ${DATA_DIR}"
echo "Out dir:  ${OUT_DIR}"
echo "Bins/min count: ${RESPONSE_N_BINS}/${RESPONSE_MIN_COUNT}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
echo "Done: ${OUT_DIR}"
