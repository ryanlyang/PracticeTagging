#!/usr/bin/env bash
#SBATCH --job-name=an18f
#SBATCH --partition=tier3
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze18_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze18_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"

M2_RUN_DIR="${M2_RUN_DIR:-${BASE_DIR}/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M3_RUN_DIR="${M3_RUN_DIR:-${BASE_DIR}/model3_recoteacher_s09/model3_recoteacher_s09_150k75k150k_seed0}"
M4_RUN_DIR="${M4_RUN_DIR:-${BASE_DIR}/model4_recoteacher_s01_corrected/model4_recoteacher_s01_corrected_150k75k150k_seed0}"
M5_RUN_DIR="${M5_RUN_DIR:-${BASE_DIR}/model5_joint_s01_full/model5_joint_s01_full_150k75k150k_seed0}"
M6_RUN_DIR="${M6_RUN_DIR:-${BASE_DIR}/model6_concat_stagea_corrected/model6_concat_stagea_corrected_150k75k150k_seed0}"

M7_RUN_DIR="${M7_RUN_DIR:-${BASE_DIR}/model7_stageA_residual_hlt/model7_stageA_residual_hlt_150k75k150k_seed0}"
M8_RUN_DIR="${M8_RUN_DIR:-${BASE_DIR}/model8_direct_residual_ablation/model8_direct_residual_ablation_150k75k150k_seed0}"
M9_LOW_RUN_DIR="${M9_LOW_RUN_DIR:-${BASE_DIR}/model9_stageA_residual_hlt_offdrop_low/model9_stageA_residual_hlt_offdrop_low_150k75k150k_seed0}"
M9_MID_RUN_DIR="${M9_MID_RUN_DIR:-${BASE_DIR}/model9_stageA_residual_hlt_offdrop_mid/model9_stageA_residual_hlt_offdrop_mid_150k75k150k_seed0}"
M9_HIGH_RUN_DIR="${M9_HIGH_RUN_DIR:-${BASE_DIR}/model9_stageA_residual_hlt_offdrop_high/model9_stageA_residual_hlt_offdrop_high_150k75k150k_seed0}"

M4_K40_RUN_DIR="${M4_K40_RUN_DIR:-${BASE_DIR}/model4_recoteacher_s01_corrected_k40/model4_recoteacher_s01_corrected_k40_150k75k150k_seed0}"
M4_K60_RUN_DIR="${M4_K60_RUN_DIR:-${BASE_DIR}/model4_recoteacher_s01_corrected_k60/model4_recoteacher_s01_corrected_k60_150k75k150k_seed0}"
M4_K80_RUN_DIR="${M4_K80_RUN_DIR:-${BASE_DIR}/model4_recoteacher_s01_corrected_k80/model4_recoteacher_s01_corrected_k80_150k75k150k_seed0}"

M10_RUN_DIR="${M10_RUN_DIR:-${BASE_DIR}/model10_recoteacher_s01_corrected_antioverlap/model10_recoteacher_s01_corrected_antioverlap_150k75k150k_seed0}"
M11_RUN_DIR="${M11_RUN_DIR:-${BASE_DIR}/model11_recoteacher_s01_corrected_feat_noangle/model11_recoteacher_s01_corrected_feat_noangle_150k75k150k_seed0}"
M12_RUN_DIR="${M12_RUN_DIR:-${BASE_DIR}/model12_recoteacher_s01_corrected_feat_noscale/model12_recoteacher_s01_corrected_feat_noscale_150k75k150k_seed0}"
M13_RUN_DIR="${M13_RUN_DIR:-${BASE_DIR}/model13_recoteacher_s01_corrected_feat_coreshape/model13_recoteacher_s01_corrected_feat_coreshape_150k75k150k_seed0}"

TARGET_TPR="${TARGET_TPR:-0.50}"
WEIGHT_STEP_2="${WEIGHT_STEP_2:-0.01}"
WEIGHT_SAMPLES_MULTI="${WEIGHT_SAMPLES_MULTI:-12000}"
PAIR_GRID_STEP_MULTI="${PAIR_GRID_STEP_MULTI:-0.05}"
META_SEL_FRAC="${META_SEL_FRAC:-0.30}"
META_C_GRID="${META_C_GRID:-0.05,0.1,0.3,1,3,10,30}"
SEED="${SEED:-0}"
OUTPUT_NAME="${OUTPUT_NAME:-fusion_hlt_joint18_analysis.json}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CMD=(
  python analyze_hlt_joint18_recoteacher_fusion.py
  --joint_delta_run_dir "${M2_RUN_DIR}"
  --reco_teacher_s09_run_dir "${M3_RUN_DIR}"
  --corrected_s01_run_dir "${M4_RUN_DIR}"
  --joint_s01_run_dir "${M5_RUN_DIR}"
  --concat_run_dir "${M6_RUN_DIR}"
  --m7_residual_run_dir "${M7_RUN_DIR}"
  --m8_direct_residual_run_dir "${M8_RUN_DIR}"
  --m9_low_run_dir "${M9_LOW_RUN_DIR}"
  --m9_mid_run_dir "${M9_MID_RUN_DIR}"
  --m9_high_run_dir "${M9_HIGH_RUN_DIR}"
  --m4_k40_run_dir "${M4_K40_RUN_DIR}"
  --m4_k60_run_dir "${M4_K60_RUN_DIR}"
  --m4_k80_run_dir "${M4_K80_RUN_DIR}"
  --m10_run_dir "${M10_RUN_DIR}"
  --m11_run_dir "${M11_RUN_DIR}"
  --m12_run_dir "${M12_RUN_DIR}"
  --m13_run_dir "${M13_RUN_DIR}"
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
echo "18-model fusion analysis"
echo "BASE_DIR: ${BASE_DIR}"
echo "M2 : ${M2_RUN_DIR}"
echo "M3 : ${M3_RUN_DIR}"
echo "M4 : ${M4_RUN_DIR}"
echo "M5 : ${M5_RUN_DIR}"
echo "M6 : ${M6_RUN_DIR}"
echo "M7 : ${M7_RUN_DIR}"
echo "M8 : ${M8_RUN_DIR}"
echo "M9L: ${M9_LOW_RUN_DIR}"
echo "M9M: ${M9_MID_RUN_DIR}"
echo "M9H: ${M9_HIGH_RUN_DIR}"
echo "K40: ${M4_K40_RUN_DIR}"
echo "K60: ${M4_K60_RUN_DIR}"
echo "K80: ${M4_K80_RUN_DIR}"
echo "M10: ${M10_RUN_DIR}"
echo "M11: ${M11_RUN_DIR}"
echo "M12: ${M12_RUN_DIR}"
echo "M13: ${M13_RUN_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done analysis: ${M2_RUN_DIR}/${OUTPUT_NAME}"
