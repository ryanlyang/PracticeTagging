#!/usr/bin/env bash
#SBATCH --job-name=jcOffDec
#SBATCH --partition=debug
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=offline_reconstructor_logs/jetclass_offline_teacher_decoder_audit_%j.out
#SBATCH --error=offline_reconstructor_logs/jetclass_offline_teacher_decoder_audit_%j.err

set -euo pipefail

REPO="${REPO:-/home/ryreu/atlas/PracticeTagging}"
cd "${REPO}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-atlas_kd}"

mkdir -p offline_reconstructor_logs

RUN_DIR="${RUN_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core11_topk60ish}"
DATA_DIR="${DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUT_DIR="${OUT_DIR:-checkpoints/jetclass_joint_dualview/fusion_reports/offline_teacher_decoder_audit_core11_topk60ish}"

N_VAL_JETS="${N_VAL_JETS:-250000}"
N_TEST_JETS="${N_TEST_JETS:-300000}"
MAX_STACK_TRAIN_ROWS="${MAX_STACK_TRAIN_ROWS:-150000}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STACK_N_JOBS="${STACK_N_JOBS:-1}"
INCLUDE_HLT_BASELINE="${INCLUDE_HLT_BASELINE:-1}"

CMD=(
  python -u analyze_jetclass_offline_teacher_decoder_audit.py
  --run_dir "${RUN_DIR}"
  --data_dir "${DATA_DIR}"
  --out_dir "${OUT_DIR}"
  --device cuda
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --max_stack_train_rows "${MAX_STACK_TRAIN_ROWS}"
  --max_eval_rows "${MAX_EVAL_ROWS}"
  --n_jobs "${STACK_N_JOBS}"
  --cv 5
  --Cs 0.03 0.1 0.3 1.0 3.0 10.0
)

if [[ "${INCLUDE_HLT_BASELINE}" == "1" ]]; then
  CMD+=(--include_hlt_baseline)
fi

echo "============================================================"
echo "JetClass Offline Teacher Decoder Audit"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Runtime: 10:00:00"
echo "Run dir: ${RUN_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Out dir: ${OUT_DIR}"
echo "Loaded val/test jets: ${N_VAL_JETS}/${N_TEST_JETS}"
echo "Stack train/eval rows: ${MAX_STACK_TRAIN_ROWS}/${MAX_EVAL_ROWS}"
echo "Include HLT baseline: ${INCLUDE_HLT_BASELINE}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
