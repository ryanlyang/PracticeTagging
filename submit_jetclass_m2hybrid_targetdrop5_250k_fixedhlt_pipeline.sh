#!/usr/bin/env bash
set -euo pipefail

# Queues five fixed-HLT JetClass m2-hybrid runs at 250k/50k/250k with only
# TARGET_DROP_PROB_MAX varied, then a dependent five-model stacked metafuser.

PARTITION="${PARTITION:-tier3}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-3-00:00:00}"
FUSION_TIME_LIMIT="${FUSION_TIME_LIMIT:-1-00:00:00}"
TRAIN_MEM="${TRAIN_MEM:-96G}"
FUSION_MEM="${FUSION_MEM:-96G}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"

RUNNER_TRAIN="${RUNNER_TRAIN:-run_train_jetclass_joint_dualview_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_targetdrop5_stacked_fusion_250k_m2hybrid_fixedhlt.sh}"

for f in "${RUNNER_TRAIN}" "${RUNNER_FUSION}"; do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

RUN000="jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop000"
RUN015="jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop015"
RUN040="jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop040"
RUN055="jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop055"
RUN070="jetclass_joint_v2attr_250k50k250k_m2hlt_hybridops_adaptivegen_fixedhlt_tdrop070"

OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/targetdrop5_250k50k250k_m2hybrid_fixedhlt_stacked_acc}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"

# Same fixed setup as the cleaned 1M m2-hybrid pipeline, just at 250k/50k/250k.
COMMON_EXPORT="SAVE_DIR=${SAVE_DIR},SEED=52,N_TRAIN_JETS=250000,N_VAL_JETS=50000,N_TEST_JETS=250000,FEATURE_PREPROCESSING=canonical,CLASS_ASSIGNMENT=filename,TARGET_CLASS=Hbb,BACKGROUND_CLASS=QCD,HLT_PT_THRESHOLD=1.30,MERGE_PROB_SCALE=1.35,REASSIGN_SCALE=1.00,SMEAR_SCALE=1.00,EFF_PLATEAU_BARREL=0.99,EFF_PLATEAU_ENDCAP=0.97,EFF_TURNON_PT=1.40,EFF_WIDTH_PT=0.20,TARGET_DROP_WARMUP_EPOCHS=20,TARGET_DROP_NUM_BANKS=3,TARGET_DROP_BANK_CYCLE_EPOCHS=1,TARGET_DROP_MODE=deterministic_bank"

submit_train() {
  local job_name="$1"
  local export_kv="$2"
  sbatch --parsable \
    --job-name="${job_name}" \
    --partition="${PARTITION}" \
    --time="${TRAIN_TIME_LIMIT}" \
    --mem="${TRAIN_MEM}" \
    --export="ALL,${COMMON_EXPORT},${export_kv}" \
    "${RUNNER_TRAIN}"
}

submit_fusion() {
  local dep="$1"
  local export_kv="$2"
  sbatch --parsable \
    --job-name="jcTD5F" \
    --partition="${PARTITION}" \
    --time="${FUSION_TIME_LIMIT}" \
    --mem="${FUSION_MEM}" \
    --dependency="${dep}" \
    --export="ALL,${export_kv}" \
    "${RUNNER_FUSION}"
}

echo "Submitting JetClass target-drop sweep at 250k/50k/250k on ${PARTITION}"
echo "Train time: ${TRAIN_TIME_LIMIT}"
echo "Fusion time: ${FUSION_TIME_LIMIT}"
echo "Only varied knob: TARGET_DROP_PROB_MAX in {0.00, 0.15, 0.40, 0.55, 0.70}"

j000=$(submit_train "jcTD000" "RUN_NAME=${RUN000},TARGET_DROP_PROB_MAX=0.00")
echo "  TDROP000 ${j000} ${RUN000}"

j015=$(submit_train "jcTD015" "RUN_NAME=${RUN015},TARGET_DROP_PROB_MAX=0.15")
echo "  TDROP015 ${j015} ${RUN015}"

j040=$(submit_train "jcTD040" "RUN_NAME=${RUN040},TARGET_DROP_PROB_MAX=0.40")
echo "  TDROP040 ${j040} ${RUN040}"

j055=$(submit_train "jcTD055" "RUN_NAME=${RUN055},TARGET_DROP_PROB_MAX=0.55")
echo "  TDROP055 ${j055} ${RUN055}"

j070=$(submit_train "jcTD070" "RUN_NAME=${RUN070},TARGET_DROP_PROB_MAX=0.70")
echo "  TDROP070 ${j070} ${RUN070}"

dep_all="afterok:${j000}:${j015}:${j040}:${j055}:${j070}"

MODEL_01_SPEC="tdrop000:stage2:${SAVE_DIR}/${RUN000}"
MODEL_02_SPEC="tdrop015:stage2:${SAVE_DIR}/${RUN015}"
MODEL_03_SPEC="tdrop040:stage2:${SAVE_DIR}/${RUN040}"
MODEL_04_SPEC="tdrop055:stage2:${SAVE_DIR}/${RUN055}"
MODEL_05_SPEC="tdrop070:stage2:${SAVE_DIR}/${RUN070}"

jfuse=$(submit_fusion "${dep_all}" \
  "SAVE_ROOT=${SAVE_DIR},OUT_DIR=${OUT_DIR},OPTIMIZE_FOR=${OPTIMIZE_FOR},MODEL_01_SPEC=${MODEL_01_SPEC},MODEL_02_SPEC=${MODEL_02_SPEC},MODEL_03_SPEC=${MODEL_03_SPEC},MODEL_04_SPEC=${MODEL_04_SPEC},MODEL_05_SPEC=${MODEL_05_SPEC}")

echo "  FUSION ${jfuse} target-drop five-model stacked metafuser"

echo "============================================================"
echo "Queued JetClass target-drop sweep"
echo "Partition:   ${PARTITION}"
echo "Train time:  ${TRAIN_TIME_LIMIT}"
echo "Fusion time: ${FUSION_TIME_LIMIT}"
echo "Train mem:   ${TRAIN_MEM}"
echo "Fusion mem:  ${FUSION_MEM}"
echo "Fusion job:  ${jfuse}"
echo "Fusion out:  ${OUT_DIR}"
echo "Dependency:  ${dep_all}"
echo "============================================================"

