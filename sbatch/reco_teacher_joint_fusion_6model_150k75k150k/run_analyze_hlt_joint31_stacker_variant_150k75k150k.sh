#!/usr/bin/env bash
#SBATCH --job-name=an31var
#SBATCH --partition=tier3
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31sweep_%x_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze31sweep_%x_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"

# Original 18-model fusion set
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

# Additional m2 delta ablations
M2_D000_RUN_DIR="${M2_D000_RUN_DIR:-${BASE_DIR}/model2_joint_delta000/model2_joint_delta000_150k75k150k_seed0}"
M2_D020_RUN_DIR="${M2_D020_RUN_DIR:-${BASE_DIR}/model2_joint_delta020/model2_joint_delta020_150k75k150k_seed0}"

# Additional dualreco family (frozen dualview score)
M11_DUAL_RUN_DIR="${M11_DUAL_RUN_DIR:-${BASE_DIR}/model11_dualreco_dualview_feat_noangle/model11_dualreco_dualview_feat_noangle_150k75k150k_seed0}"
M12_DUAL_RUN_DIR="${M12_DUAL_RUN_DIR:-${BASE_DIR}/model12_dualreco_dualview_feat_noscale/model12_dualreco_dualview_feat_noscale_150k75k150k_seed0}"
M13_DUAL_RUN_DIR="${M13_DUAL_RUN_DIR:-${BASE_DIR}/model13_dualreco_dualview_feat_coreshape/model13_dualreco_dualview_feat_coreshape_150k75k150k_seed0}"

M15_DUAL_LOW_RUN_DIR="${M15_DUAL_LOW_RUN_DIR:-${BASE_DIR}/model15_dualreco_dualview_offdrop_low/model15_dualreco_dualview_offdrop_low_150k75k150k_seed0}"
M15_DUAL_MID_RUN_DIR="${M15_DUAL_MID_RUN_DIR:-${BASE_DIR}/model15_dualreco_dualview_offdrop_mid/model15_dualreco_dualview_offdrop_mid_150k75k150k_seed0}"
M15_DUAL_HIGH_RUN_DIR="${M15_DUAL_HIGH_RUN_DIR:-${BASE_DIR}/model15_dualreco_dualview_offdrop_high/model15_dualreco_dualview_offdrop_high_150k75k150k_seed0}"

M16_DUAL_K40_RUN_DIR="${M16_DUAL_K40_RUN_DIR:-${BASE_DIR}/model16_dualreco_dualview_topk40/model16_dualreco_dualview_topk40_150k75k150k_seed0}"
M16_DUAL_K60_RUN_DIR="${M16_DUAL_K60_RUN_DIR:-${BASE_DIR}/model16_dualreco_dualview_topk60/model16_dualreco_dualview_topk60_150k75k150k_seed0}"
M16_DUAL_K80_RUN_DIR="${M16_DUAL_K80_RUN_DIR:-${BASE_DIR}/model16_dualreco_dualview_topk80/model16_dualreco_dualview_topk80_150k75k150k_seed0}"

M17_DUAL_RUN_DIR="${M17_DUAL_RUN_DIR:-${BASE_DIR}/model17_dualreco_dualview_antioverlap/model17_dualreco_dualview_antioverlap_150k75k150k_seed0}"
M19_DUAL_RUN_DIR="${M19_DUAL_RUN_DIR:-${BASE_DIR}/model19_dualreco_dualview_basic/model19_dualreco_dualview_basic_150k75k150k_seed0}"

TARGET_TPR="${TARGET_TPR:-0.50}"
WEIGHT_STEP_2="${WEIGHT_STEP_2:-0.01}"
WEIGHT_SAMPLES_MULTI="${WEIGHT_SAMPLES_MULTI:-12000}"
WEIGHT_SAMPLES_MULTI_SPARSE="${WEIGHT_SAMPLES_MULTI_SPARSE:-20000}"
WEIGHT_SPARSE_K_GRID="${WEIGHT_SPARSE_K_GRID:-2,3,4,5,6,8,10,12}"
PAIR_GRID_STEP_MULTI="${PAIR_GRID_STEP_MULTI:-0.05}"
META_SEL_FRAC="${META_SEL_FRAC:-0.30}"
META_C_GRID="${META_C_GRID:-0.05,0.1,0.3,1,3,10,30}"
SEED="${SEED:-0}"

# Optional advanced-parameter overrides
SPARSE_L1_GRID="${SPARSE_L1_GRID:-}"
SPARSE_FOLDS="${SPARSE_FOLDS:-}"
SPARSE_STABILITY_SEEDS="${SPARSE_STABILITY_SEEDS:-}"
SPARSE_MIN_FREQ="${SPARSE_MIN_FREQ:-}"
SPARSE_SELECT_THRESHOLD="${SPARSE_SELECT_THRESHOLD:-}"
SPARSE_TRAIN_STEPS="${SPARSE_TRAIN_STEPS:-}"
SPARSE_LR="${SPARSE_LR:-}"

MLP_HIDDEN_GRID="${MLP_HIDDEN_GRID:-}"
MLP_ALPHA_GRID="${MLP_ALPHA_GRID:-}"
MLP_MAX_ITER="${MLP_MAX_ITER:-}"

MOE_ENTROPY_GRID="${MOE_ENTROPY_GRID:-}"
MOE_L2_GRID="${MOE_L2_GRID:-}"
MOE_TEMP_GRID="${MOE_TEMP_GRID:-}"
MOE_TOPK_GRID="${MOE_TOPK_GRID:-}"
MOE_STEPS="${MOE_STEPS:-}"
MOE_BATCH_SIZE="${MOE_BATCH_SIZE:-}"
MOE_LR="${MOE_LR:-}"

FUSION_VARIANT="${FUSION_VARIANT:-all_stackers}"
OUTPUT_NAME="${OUTPUT_NAME:-fusion_hlt_joint31_${FUSION_VARIANT}.json}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CMD=(
  python analyze_hlt_joint35_recoteacher_fusion.py
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

  --m2_delta000_run_dir "${M2_D000_RUN_DIR}"
  --m2_delta020_run_dir "${M2_D020_RUN_DIR}"

  --m11_dual_run_dir "${M11_DUAL_RUN_DIR}"
  --m12_dual_run_dir "${M12_DUAL_RUN_DIR}"
  --m13_dual_run_dir "${M13_DUAL_RUN_DIR}"
  --m15_dual_low_run_dir "${M15_DUAL_LOW_RUN_DIR}"
  --m15_dual_mid_run_dir "${M15_DUAL_MID_RUN_DIR}"
  --m15_dual_high_run_dir "${M15_DUAL_HIGH_RUN_DIR}"
  --m16_dual_k40_run_dir "${M16_DUAL_K40_RUN_DIR}"
  --m16_dual_k60_run_dir "${M16_DUAL_K60_RUN_DIR}"
  --m16_dual_k80_run_dir "${M16_DUAL_K80_RUN_DIR}"
  --m17_dual_run_dir "${M17_DUAL_RUN_DIR}"
  --m19_dual_run_dir "${M19_DUAL_RUN_DIR}"

  --target_tpr "${TARGET_TPR}"
  --weight_step_2 "${WEIGHT_STEP_2}"
  --weight_samples_multi "${WEIGHT_SAMPLES_MULTI}"
  --weight_samples_multi_sparse "${WEIGHT_SAMPLES_MULTI_SPARSE}"
  --weight_sparse_k_grid "${WEIGHT_SPARSE_K_GRID}"
  --pair_grid_step_multi "${PAIR_GRID_STEP_MULTI}"
  --meta_sel_frac "${META_SEL_FRAC}"
  --meta_c_grid "${META_C_GRID}"
  --seed "${SEED}"
  --output_name "${OUTPUT_NAME}"
)

# Optional advanced overrides
[[ -n "${SPARSE_L1_GRID}" ]] && CMD+=(--sparse_l1_grid "${SPARSE_L1_GRID}")
[[ -n "${SPARSE_FOLDS}" ]] && CMD+=(--sparse_folds "${SPARSE_FOLDS}")
[[ -n "${SPARSE_STABILITY_SEEDS}" ]] && CMD+=(--sparse_stability_seeds "${SPARSE_STABILITY_SEEDS}")
[[ -n "${SPARSE_MIN_FREQ}" ]] && CMD+=(--sparse_min_freq "${SPARSE_MIN_FREQ}")
[[ -n "${SPARSE_SELECT_THRESHOLD}" ]] && CMD+=(--sparse_select_threshold "${SPARSE_SELECT_THRESHOLD}")
[[ -n "${SPARSE_TRAIN_STEPS}" ]] && CMD+=(--sparse_train_steps "${SPARSE_TRAIN_STEPS}")
[[ -n "${SPARSE_LR}" ]] && CMD+=(--sparse_lr "${SPARSE_LR}")

[[ -n "${MLP_HIDDEN_GRID}" ]] && CMD+=(--mlp_hidden_grid "${MLP_HIDDEN_GRID}")
[[ -n "${MLP_ALPHA_GRID}" ]] && CMD+=(--mlp_alpha_grid "${MLP_ALPHA_GRID}")
[[ -n "${MLP_MAX_ITER}" ]] && CMD+=(--mlp_max_iter "${MLP_MAX_ITER}")

[[ -n "${MOE_ENTROPY_GRID}" ]] && CMD+=(--moe_entropy_grid "${MOE_ENTROPY_GRID}")
[[ -n "${MOE_L2_GRID}" ]] && CMD+=(--moe_l2_grid "${MOE_L2_GRID}")
[[ -n "${MOE_TEMP_GRID}" ]] && CMD+=(--moe_temp_grid "${MOE_TEMP_GRID}")
[[ -n "${MOE_TOPK_GRID}" ]] && CMD+=(--moe_topk_grid "${MOE_TOPK_GRID}")
[[ -n "${MOE_STEPS}" ]] && CMD+=(--moe_steps "${MOE_STEPS}")
[[ -n "${MOE_BATCH_SIZE}" ]] && CMD+=(--moe_batch_size "${MOE_BATCH_SIZE}")
[[ -n "${MOE_LR}" ]] && CMD+=(--moe_lr "${MOE_LR}")

# Variant toggles
case "${FUSION_VARIANT}" in
  baseline_meta)
    CMD+=(--disable_sparse_stacker --disable_tiny_mlp_stacker --disable_moe_gated_stacker)
    ;;
  sparse_only)
    CMD+=(--disable_tiny_mlp_stacker --disable_moe_gated_stacker)
    ;;
  tiny_mlp_only)
    CMD+=(--disable_sparse_stacker --disable_moe_gated_stacker)
    ;;
  moe_only)
    CMD+=(--disable_sparse_stacker --disable_tiny_mlp_stacker)
    ;;
  sparse_mlp)
    CMD+=(--disable_moe_gated_stacker)
    ;;
  sparse_moe)
    CMD+=(--disable_tiny_mlp_stacker)
    ;;
  mlp_moe)
    CMD+=(--disable_sparse_stacker)
    ;;
  all_stackers)
    ;;
  *)
    echo "Unknown FUSION_VARIANT='${FUSION_VARIANT}'"
    echo "Valid: baseline_meta, sparse_only, tiny_mlp_only, moe_only, sparse_mlp, sparse_moe, mlp_moe, all_stackers"
    exit 2
    ;;
esac

echo "============================================================"
echo "31-model fusion stacker sweep variant"
echo "Variant: ${FUSION_VARIANT}"
echo "Output:  ${OUTPUT_NAME}"
echo "BASE_DIR: ${BASE_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done analysis: ${M2_RUN_DIR}/${OUTPUT_NAME}"
