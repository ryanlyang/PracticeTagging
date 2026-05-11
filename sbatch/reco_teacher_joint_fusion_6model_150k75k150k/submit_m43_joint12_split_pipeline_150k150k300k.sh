#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ANALYZE="${HERE}/run_analyze_hlt_joint12_bin_gated_fusion_valsel_weighted_150k150k300k.sh"
RUN_BUILD="${HERE}/run_build_joint12_fused_targets_weighted_150k150k300k.sh"
RUN_CHECK="${HERE}/run_check_joint12_fused_targets_split_150k150k300k.sh"
RUN_M43="${HERE}/run_m43_joint12_fusedadv_stagea_dualkd_weighted_150k150k300k.sh"

BASE_CKPT="${BASE_CKPT:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
ANALYZE_OUT_DIR="${ANALYZE_OUT_DIR:-${BASE_CKPT}/bin_gated_fusion_12_weighted_150k150k300k_split}"
ANALYZE_FUSION_JSON="${ANALYZE_FUSION_JSON:-${ANALYZE_OUT_DIR}/fusion_hlt_joint12_weighted_150k150k300k_split.json}"
TARGETS_OUT_DIR="${TARGETS_OUT_DIR:-${BASE_CKPT}/fused_targets_joint12_weighted_150k150k300k_split}"
TARGETS_NPZ="${TARGETS_NPZ:-${TARGETS_OUT_DIR}/fused_targets_train_val_test.npz}"
SCORES_NPZ="${SCORES_NPZ:-${ANALYZE_OUT_DIR}/bin_gated_scores.npz}"

M43_SAVE_DIR="${M43_SAVE_DIR:-${BASE_CKPT}/model43_joint12_fusedadv_stagea_dualkd_v2_weighted_150k150k300k_splittargets}"
M43_RUN_NAME="${M43_RUN_NAME:-model43_joint12_fusedadv_stagea_dualkd_v2_weighted_150k150k300k_splittargets_seed0}"

M43_STAGEA_FUSED_UNCERT_WEIGHT="${M43_STAGEA_FUSED_UNCERT_WEIGHT:-0.50}"
M43_STAGEA_FUSED_KD_W_MAX="${M43_STAGEA_FUSED_KD_W_MAX:-4.00}"
M43_JOINT_FUSED_KD_W_MAX="${M43_JOINT_FUSED_KD_W_MAX:-3.00}"
M43_STAGEC_JOINT_FUSED_KD_LAMBDA="${M43_STAGEC_JOINT_FUSED_KD_LAMBDA:-0.28}"

SELECTION_MODE="${SELECTION_MODE:-split}"
DEP_JOB_ID="${DEP_JOB_ID:-}"

dep_flag=()
if [[ -n "${DEP_JOB_ID}" ]]; then
  dep_flag=(--dependency="afterok:${DEP_JOB_ID}")
fi

j_analyze=$(
  sbatch --parsable "${dep_flag[@]}" \
    --export="ALL,SELECTION_MODE=${SELECTION_MODE},OUT_DIR=${ANALYZE_OUT_DIR},FUSION_JSON=${ANALYZE_FUSION_JSON}" \
    "${RUN_ANALYZE}"
)

j_build=$(
  sbatch --parsable --dependency="afterok:${j_analyze}" \
    --export="ALL,SCORES_NPZ=${SCORES_NPZ},OUT_DIR=${TARGETS_OUT_DIR}" \
    "${RUN_BUILD}"
)

j_check=$(
  sbatch --parsable --dependency="afterok:${j_build}" \
    --export="ALL,FUSED_TARGETS_NPZ=${TARGETS_NPZ}" \
    "${RUN_CHECK}"
)

j_m43=$(
  sbatch --parsable --dependency="afterok:${j_check}" \
    --export="ALL,FUSED_TARGETS_NPZ=${TARGETS_NPZ},SAVE_DIR=${M43_SAVE_DIR},RUN_NAME=${M43_RUN_NAME},STAGEA_FUSED_UNCERT_WEIGHT=${M43_STAGEA_FUSED_UNCERT_WEIGHT},STAGEA_FUSED_KD_W_MAX=${M43_STAGEA_FUSED_KD_W_MAX},JOINT_FUSED_KD_W_MAX=${M43_JOINT_FUSED_KD_W_MAX},STAGEC_JOINT_FUSED_KD_LAMBDA=${M43_STAGEC_JOINT_FUSED_KD_LAMBDA}" \
    "${RUN_M43}"
)

echo "Submitted split-safe m43 pipeline:"
echo "  analyze_split=${j_analyze}"
echo "  build_targets=${j_build} (afterok:${j_analyze})"
echo "  check_split=${j_check} (afterok:${j_build})"
echo "  m43=${j_m43} (afterok:${j_check})"
echo
echo "Artifacts:"
echo "  analyze_out_dir=${ANALYZE_OUT_DIR}"
echo "  scores_npz=${SCORES_NPZ}"
echo "  targets_npz=${TARGETS_NPZ}"
echo "  m43_save_dir=${M43_SAVE_DIR}/${M43_RUN_NAME}"
