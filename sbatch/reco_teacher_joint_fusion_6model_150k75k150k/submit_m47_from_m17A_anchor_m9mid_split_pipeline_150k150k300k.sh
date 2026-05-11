#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ANALYZE="${HERE}/run_analyze_hlt_joint12_bin_gated_fusion_valsel_weighted_150k150k300k.sh"
RUN_BUILD="${HERE}/run_build_joint12_fused_targets_weighted_150k150k300k.sh"
RUN_CHECK="${HERE}/run_check_joint12_fused_targets_split_150k150k300k.sh"
RUN_M47="${HERE}/run_m47_joint12_fuseddelta_from_m17A_anchor_m9mid_split_weighted_150k150k300k.sh"

BASE_CKPT="${BASE_CKPT:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
ANALYZE_OUT_DIR="${ANALYZE_OUT_DIR:-${BASE_CKPT}/bin_gated_fusion_12_weighted_150k150k300k_split_m47}"
ANALYZE_FUSION_JSON="${ANALYZE_FUSION_JSON:-${ANALYZE_OUT_DIR}/fusion_hlt_joint12_weighted_150k150k300k_split_m47.json}"
TARGETS_OUT_DIR="${TARGETS_OUT_DIR:-${BASE_CKPT}/fused_targets_joint12_weighted_150k150k300k_split_m47}"
TARGETS_NPZ="${TARGETS_NPZ:-${TARGETS_OUT_DIR}/fused_targets_train_val_test.npz}"
SCORES_NPZ="${SCORES_NPZ:-${ANALYZE_OUT_DIR}/bin_gated_scores.npz}"

M47_SAVE_DIR="${M47_SAVE_DIR:-${BASE_CKPT}/model47_from_m17A_anchor_m9mid_split_weighted_150k150k300k}"
M47_RUN_NAME="${M47_RUN_NAME:-model47_from_m17A_anchor_m9mid_split_weighted_150k150k300k_seed0}"
M47_STAGEA_LOAD_CKPT="${M47_STAGEA_LOAD_CKPT:-${BASE_CKPT}/model17_dualreco_dualview_antioverlap_weighted_150k150k300k/model17_dualreco_dualview_antioverlap_weighted_150k150k300k_seed0/offline_reconstructor_A_stageA.pt}"
M47_ANCHOR_RECO_CKPT="${M47_ANCHOR_RECO_CKPT:-${BASE_CKPT}/model9_stageA_residual_hlt_offdrop_mid_weighted_150k150k300k/model9_stageA_residual_hlt_offdrop_mid_weighted_150k150k300k_seed0/offline_reconstructor_stageA.pt}"
M47_SOURCE_SPLITS_NPZ="${M47_SOURCE_SPLITS_NPZ:-${BASE_CKPT}/model2_joint_delta005_weighted_150k150k300k/model2_joint_delta005_weighted_150k150k300k_seed0/data_splits.npz}"
M47_JOINT_EPOCHS="${M47_JOINT_EPOCHS:-36}"
M47_JOINT_PATIENCE="${M47_JOINT_PATIENCE:-14}"

DEP_JOB_ID="${DEP_JOB_ID:-}"
dep_flag=()
if [[ -n "${DEP_JOB_ID}" ]]; then
  dep_flag=(--dependency="afterok:${DEP_JOB_ID}")
fi

j_analyze=$(
  sbatch --parsable "${dep_flag[@]}" \
    --export="ALL,SELECTION_MODE=split,OUT_DIR=${ANALYZE_OUT_DIR},FUSION_JSON=${ANALYZE_FUSION_JSON}" \
    "${RUN_ANALYZE}"
)

j_build=$(
  sbatch --parsable --dependency="afterok:${j_analyze}" \
    --export="ALL,SCORES_NPZ=${SCORES_NPZ},OUT_DIR=${TARGETS_OUT_DIR},ALLOW_FIT_REF_OVERLAP=0" \
    "${RUN_BUILD}"
)

j_check=$(
  sbatch --parsable --dependency="afterok:${j_build}" \
    --export="ALL,FUSED_TARGETS_NPZ=${TARGETS_NPZ}" \
    "${RUN_CHECK}"
)

j_m47=$(
  sbatch --parsable --dependency="afterok:${j_check}" \
    --export="ALL,FUSED_TARGETS_NPZ=${TARGETS_NPZ},FUSED_SOURCE_SPLITS_NPZ=${M47_SOURCE_SPLITS_NPZ},RESIDUAL_SELECT_METRIC=auc,RESIDUAL_TRAIN_FROM=fit,RESIDUAL_VAL_FROM=ref,RESIDUAL_TEST_FROM=source_test,STAGEA_LOAD_CKPT=${M47_STAGEA_LOAD_CKPT},ANCHOR_LOGIT_SOURCE=reco_teacher,ANCHOR_RECO_CKPT=${M47_ANCHOR_RECO_CKPT},ANCHOR_TEACHER_SOURCE=teacher,SAVE_DIR=${M47_SAVE_DIR},RUN_NAME=${M47_RUN_NAME},RESIDUAL_JOINT_EPOCHS=${M47_JOINT_EPOCHS},RESIDUAL_JOINT_PATIENCE=${M47_JOINT_PATIENCE}" \
    "${RUN_M47}"
)

echo "Submitted split-safe m47 pipeline:"
echo "  analyze_split=${j_analyze}"
echo "  build_targets=${j_build} (afterok:${j_analyze})"
echo "  check_split=${j_check} (afterok:${j_build})"
echo "  m47=${j_m47} (afterok:${j_check})"
echo
echo "Artifacts:"
echo "  analyze_out_dir=${ANALYZE_OUT_DIR}"
echo "  scores_npz=${SCORES_NPZ}"
echo "  targets_npz=${TARGETS_NPZ}"
echo "  m47_save_dir=${M47_SAVE_DIR}/${M47_RUN_NAME}"
