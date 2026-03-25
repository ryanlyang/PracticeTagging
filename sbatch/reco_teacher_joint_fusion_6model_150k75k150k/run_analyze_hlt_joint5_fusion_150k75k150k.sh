#!/usr/bin/env bash
#SBATCH --job-name=an5f150
#SBATCH --partition=tier3
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze5_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze5_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
M2_RUN_DIR="${M2_RUN_DIR:-${BASE_DIR}/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M3_RUN_DIR="${M3_RUN_DIR:-${BASE_DIR}/model3_recoteacher_s09/model3_recoteacher_s09_150k75k150k_seed0}"
M4_RUN_DIR="${M4_RUN_DIR:-${BASE_DIR}/model4_recoteacher_s01_corrected/model4_recoteacher_s01_corrected_150k75k150k_seed0}"
M5_RUN_DIR="${M5_RUN_DIR:-${BASE_DIR}/model5_joint_s01_full/model5_joint_s01_full_150k75k150k_seed0}"

TARGET_TPR="${TARGET_TPR:-0.50}"
WEIGHT_STEP_2="${WEIGHT_STEP_2:-0.01}"
WEIGHT_SAMPLES_MULTI="${WEIGHT_SAMPLES_MULTI:-4000}"
PAIR_GRID_STEP_MULTI="${PAIR_GRID_STEP_MULTI:-0.05}"
META_SEL_FRAC="${META_SEL_FRAC:-0.30}"
META_C_GRID="${META_C_GRID:-0.05,0.1,0.3,1,3,10,30}"
SEED="${SEED:-0}"
OUTPUT_NAME="${OUTPUT_NAME:-fusion_hlt_joint5_analysis.json}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CMD=(
  python analyze_hlt_joint5_recoteacher_fusion.py
  --joint_delta_run_dir "${M2_RUN_DIR}"
  --reco_teacher_s09_run_dir "${M3_RUN_DIR}"
  --corrected_s01_run_dir "${M4_RUN_DIR}"
  --joint_s01_run_dir "${M5_RUN_DIR}"
  --target_tpr "${TARGET_TPR}"
  --weight_step_2 "${WEIGHT_STEP_2}"
  --weight_samples_multi "${WEIGHT_SAMPLES_MULTI}"
  --pair_grid_step_multi "${PAIR_GRID_STEP_MULTI}"
  --meta_sel_frac "${META_SEL_FRAC}"
  --meta_c_grid "${META_C_GRID}"
  --seed "${SEED}"
  --output_name "${OUTPUT_NAME}"
)

echo "============================================================"
echo "Five-model fusion analysis (HLT + m2/m3/m4/m5)"
echo "M2: ${M2_RUN_DIR}"
echo "M3: ${M3_RUN_DIR}"
echo "M4: ${M4_RUN_DIR}"
echo "M5: ${M5_RUN_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done analysis: ${M2_RUN_DIR}/${OUTPUT_NAME}"
