#!/usr/bin/env bash
#SBATCH --job-name=m35hyb
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m35_hybrid_m33m34_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m35_hybrid_m33m34_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

RUN_NAME="${RUN_NAME:-model35_hybrid_m33m34_150k75k150k_seed0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model35_hybrid_m33m34}"
M33_RUN_DIR="${M33_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model33_detfeas_dualview_postrefine/model33_k6_detfeas_dualview_postrefine_150k75k150k_seed0}"
M34_RUN_DIR="${M34_RUN_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model34_globalcand_multiview/model34_k12_globalcand_multiview3_150k75k150k_seed0}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-6}"

N_TRAIN_JETS="${N_TRAIN_JETS:-375000}"
N_TRAIN_SPLIT="${N_TRAIN_SPLIT:-100000}"
N_VAL_SPLIT="${N_VAL_SPLIT:-75000}"
N_TEST_SPLIT="${N_TEST_SPLIT:-150000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-80}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"

mkdir -p "${SAVE_DIR}"

CMD=(
  python offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_m35_hybrid_m33m34.py
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --m33_run_dir "${M33_RUN_DIR}"
  --m34_run_dir "${M34_RUN_DIR}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_train_split "${N_TRAIN_SPLIT}"
  --n_val_split "${N_VAL_SPLIT}"
  --n_test_split "${N_TEST_SPLIT}"
  --offset_jets "${OFFSET_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --m33_search_target_k 6
  --m34_search_target_k 12
  --m34_mv_n_select 2
)

echo "============================================================"
echo "Model-35 Hybrid TopTagger (m33 + m34 non-top-tagger stacks)"
echo "Run: ${SAVE_DIR}/${RUN_NAME}"
echo "m33_run: ${M33_RUN_DIR}"
echo "m34_run: ${M34_RUN_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done: ${SAVE_DIR}/${RUN_NAME}"
