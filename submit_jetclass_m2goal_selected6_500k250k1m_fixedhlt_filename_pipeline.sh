#!/usr/bin/env bash
set -euo pipefail

# Queues six selected fixed-HLT m2-goal variants, then a dependent stacked
# metafuser. Selected models: antioverlap, budgetlite, offdrophigh,
# reassignstrong, splitlight, topk60ish.

PARTITION="${PARTITION:-tier3}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-${TIME_LIMIT:-2-00:00:00}}"
FUSION_TIME_LIMIT="${FUSION_TIME_LIMIT:-1-00:00:00}"
TRAIN_MEM="${TRAIN_MEM:-256G}"
FUSION_MEM="${FUSION_MEM:-128G}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
TAG="${TAG:-fixedhlt}"

RUNNER_TRAIN="${RUNNER_TRAIN:-run_train_jetclass_joint_dualview_v2attr_500k250k1m_m2hlt_hybridops_goal.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_six_model_stacked_fusion_500k250k1m_m2goal_selected6.sh}"

for f in "${RUNNER_TRAIN}" "${RUNNER_FUSION}"; do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

CORE03="jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_${TAG}_core03_budgetlite"
CORE07="jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_${TAG}_core07_splitlight"
CORE08="jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_${TAG}_core08_reassignstrong"
CORE10="jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_${TAG}_core10_offdrophigh"
CORE11="jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_${TAG}_core11_topk60ish"
CORE12="jetclass_joint_v2attr_500k250k1m_m2hlt_hybridops_goal_${TAG}_core12_antioverlap"

OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/six_model_500k250k1m_m2goal_selected6_${TAG}_stacked_acc}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"

COMMON_EXPORT="SAVE_DIR=${SAVE_DIR},SEED=52,N_TRAIN_JETS=500000,N_VAL_JETS=250000,N_TEST_JETS=1000000,FEATURE_PREPROCESSING=canonical,CLASS_ASSIGNMENT=filename,TARGET_CLASS=Hbb,BACKGROUND_CLASS=QCD,HLT_PT_THRESHOLD=1.30,MERGE_PROB_SCALE=1.35,REASSIGN_SCALE=1.00,SMEAR_SCALE=1.00,EFF_PLATEAU_BARREL=0.99,EFF_PLATEAU_ENDCAP=0.97,EFF_TURNON_PT=1.40,EFF_WIDTH_PT=0.20"

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
    --job-name="jc6FG" \
    --partition="${PARTITION}" \
    --time="${FUSION_TIME_LIMIT}" \
    --mem="${FUSION_MEM}" \
    --dependency="${dep}" \
    --export="ALL,${export_kv}" \
    "${RUNNER_FUSION}"
}

echo "Submitting selected-6 fixed-HLT JetClass m2-goal 500k/250k/1M pipeline on ${PARTITION}"
echo "Train time/mem:  ${TRAIN_TIME_LIMIT} / ${TRAIN_MEM}"
echo "Fusion time/mem: ${FUSION_TIME_LIMIT} / ${FUSION_MEM}"
echo "Fixed HLT: pt_thr=1.30 merge=1.35 reassign=1.00 smear=1.00 eff=0.99/0.97 turnon=1.40 width=0.20"
echo "Selected: antioverlap, budgetlite, offdrophigh, reassignstrong, splitlight, topk60ish"

j12=$(submit_train "jcG12" "RUN_NAME=${CORE12},LOSS_W_LOCAL=0.14,LOSS_GEN_LOCAL_RADIUS=0.06,LOSS_W_SPARSE=0.013,GOAL_LAMBDA_AXIS=0.20")
echo "  m2_antioverlap   ${j12} ${CORE12}"

j03=$(submit_train "jcG03" "RUN_NAME=${CORE03},LOSS_W_BUDGET=0.55,LOSS_W_PT_RATIO=0.18,LOSS_W_SPARSE=0.012")
echo "  m2_budgetlite    ${j03} ${CORE03}"

j10=$(submit_train "jcG10" "RUN_NAME=${CORE10},TARGET_DROP_PROB_MAX=0.70,TARGET_DROP_WARMUP_EPOCHS=20,TARGET_DROP_NUM_BANKS=3,TARGET_DROP_BANK_CYCLE_EPOCHS=1")
echo "  m2_offdrophigh   ${j10} ${CORE10}"

j08=$(submit_train "jcG08" "RUN_NAME=${CORE08},LOSS_W_PT_RATIO=0.22,LOSS_W_M_RATIO=0.05,LOSS_W_E_RATIO=0.05,GOAL_LAMBDA_RESPONSE=0.45,GOAL_LAMBDA_AXIS=0.15")
echo "  m2_reassignstrong ${j08} ${CORE08}"

j07=$(submit_train "jcG07" "RUN_NAME=${CORE07},LOSS_W_LOCAL=0.05,LOSS_W_BUDGET=0.65")
echo "  m2_splitlight    ${j07} ${CORE07}"

j11=$(submit_train "jcG11" "RUN_NAME=${CORE11},RECO_MAX_GENERATED_TOKENS=60,LOSS_W_BUDGET=0.75,LOSS_W_SPARSE=0.009")
echo "  m2_topk60ish     ${j11} ${CORE11}"

dep_all="afterok:${j12}:${j03}:${j10}:${j08}:${j07}:${j11}"

MODEL_01_SPEC="m2_antioverlap:stage2:${SAVE_DIR}/${CORE12}"
MODEL_02_SPEC="m2_budgetlite:stage2:${SAVE_DIR}/${CORE03}"
MODEL_03_SPEC="m2_offdrophigh:stage2:${SAVE_DIR}/${CORE10}"
MODEL_04_SPEC="m2_reassignstrong:stage2:${SAVE_DIR}/${CORE08}"
MODEL_05_SPEC="m2_splitlight:stage2:${SAVE_DIR}/${CORE07}"
MODEL_06_SPEC="m2_topk60ish:stage2:${SAVE_DIR}/${CORE11}"

jfuse=$(submit_fusion "${dep_all}" \
  "SAVE_ROOT=${SAVE_DIR},TAG=${TAG},OUT_DIR=${OUT_DIR},OPTIMIZE_FOR=${OPTIMIZE_FOR},MODEL_01_SPEC=${MODEL_01_SPEC},MODEL_02_SPEC=${MODEL_02_SPEC},MODEL_03_SPEC=${MODEL_03_SPEC},MODEL_04_SPEC=${MODEL_04_SPEC},MODEL_05_SPEC=${MODEL_05_SPEC},MODEL_06_SPEC=${MODEL_06_SPEC}")

echo "  FUSION ${jfuse} selected-6 stacked metafuser"

echo "============================================================"
echo "Queued selected-6 fixed-HLT JetClass m2-goal pipeline"
echo "Fusion job: ${jfuse}"
echo "Fusion out: ${OUT_DIR}"
echo "Dependency: ${dep_all}"
echo "============================================================"
