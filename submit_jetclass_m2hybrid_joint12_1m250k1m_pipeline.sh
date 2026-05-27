#!/usr/bin/env bash
set -euo pipefail

# Queues 12 JetClass m2-hybrid adaptive-generation variants at 1M/250k/1M,
# then a dependent 12-model stacked metafuser (optimize_for=acc).

PARTITION="${PARTITION:-tier3}"
TIME_LIMIT="${TIME_LIMIT:-5-00:00:00}"
TRAIN_MEM="${TRAIN_MEM:-256G}"
FUSION_MEM="${FUSION_MEM:-160G}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"

RUNNER_TRAIN="${RUNNER_TRAIN:-run_train_jetclass_joint_dualview_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_twelve_model_stacked_fusion_1m250k1m_m2hybrid.sh}"

for f in "${RUNNER_TRAIN}" "${RUNNER_FUSION}"; do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

CORE01="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core01_base"
CORE02="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core02_consstrong"
CORE03="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core03_budgetlite"
CORE04="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core04_genlow"
CORE05="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core05_genhigh"
CORE06="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core06_splitstrong"
CORE07="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core07_splitlight"
CORE08="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core08_reassignstrong"
CORE09="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core09_offdropmid"
CORE10="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core10_offdrophigh"
CORE11="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core11_topk60ish"
CORE12="jetclass_joint_v2attr_1m250k1m_m2hlt_hybridops_adaptivegen_core12_antioverlap"

OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/twelve_model_1m250k1m_m2hybrid_stacked_acc}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"

COMMON_EXPORT="SAVE_DIR=${SAVE_DIR},N_TRAIN_JETS=1000000,N_VAL_JETS=250000,N_TEST_JETS=1000000,FEATURE_PREPROCESSING=canonical,CLASS_ASSIGNMENT=canonical_labels,TARGET_CLASS=Hbb,BACKGROUND_CLASS=QCD"

submit_train() {
  local job_name="$1"
  local export_kv="$2"
  local mem_args=()
  if [[ -n "${TRAIN_MEM}" ]]; then
    mem_args=(--mem="${TRAIN_MEM}")
  fi
  sbatch --parsable \
    --job-name="${job_name}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    "${mem_args[@]}" \
    --export="ALL,${COMMON_EXPORT},${export_kv}" \
    "${RUNNER_TRAIN}"
}

submit_fusion() {
  local dep="$1"
  local export_kv="$2"
  local mem_args=()
  if [[ -n "${FUSION_MEM}" ]]; then
    mem_args=(--mem="${FUSION_MEM}")
  fi
  sbatch --parsable \
    --job-name="jc12F1M" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    "${mem_args[@]}" \
    --dependency="${dep}" \
    --export="ALL,${export_kv}" \
    "${RUNNER_FUSION}"
}

echo "Submitting JetClass 12-runner m2-hybrid 1M/250k/1M pipeline on ${PARTITION} (time=${TIME_LIMIT})"

j01=$(submit_train "jcM2A01" "RUN_NAME=${CORE01}")
echo "  CORE01 ${j01} ${CORE01}"

j02=$(submit_train "jcM2A02" "RUN_NAME=${CORE02},LAMBDA_CONS=0.06,LOSS_W_BUDGET=0.80,STAGEC_LR_DUAL=1e-4,STAGEC_LR_RECO=5e-5")
echo "  CORE02 ${j02} ${CORE02}"

j03=$(submit_train "jcM2A03" "RUN_NAME=${CORE03},LOSS_W_BUDGET=0.55,LOSS_W_PT_RATIO=0.18,LOSS_W_SPARSE=0.012")
echo "  CORE03 ${j03} ${CORE03}"

j04=$(submit_train "jcM2A04" "RUN_NAME=${CORE04},RECO_MAX_GENERATED_TOKENS=40,LOSS_W_SPARSE=0.012")
echo "  CORE04 ${j04} ${CORE04}"

j05=$(submit_train "jcM2A05" "RUN_NAME=${CORE05},RECO_MAX_GENERATED_TOKENS=72,LOSS_W_SPARSE=0.006")
echo "  CORE05 ${j05} ${CORE05}"

j06=$(submit_train "jcM2A06" "RUN_NAME=${CORE06},MERGE_PROB_SCALE=1.50,LOSS_W_LOCAL=0.12,LOSS_W_BUDGET=0.75")
echo "  CORE06 ${j06} ${CORE06}"

j07=$(submit_train "jcM2A07" "RUN_NAME=${CORE07},MERGE_PROB_SCALE=1.20,LOSS_W_LOCAL=0.05,LOSS_W_BUDGET=0.65")
echo "  CORE07 ${j07} ${CORE07}"

j08=$(submit_train "jcM2A08" "RUN_NAME=${CORE08},REASSIGN_SCALE=1.20,SMEAR_SCALE=1.10")
echo "  CORE08 ${j08} ${CORE08}"

j09=$(submit_train "jcM2A09" "RUN_NAME=${CORE09},EFF_PLATEAU_BARREL=0.97,EFF_PLATEAU_ENDCAP=0.93,EFF_WIDTH_PT=0.26")
echo "  CORE09 ${j09} ${CORE09}"

j10=$(submit_train "jcM2A10" "RUN_NAME=${CORE10},EFF_PLATEAU_BARREL=0.95,EFF_PLATEAU_ENDCAP=0.88,EFF_WIDTH_PT=0.32")
echo "  CORE10 ${j10} ${CORE10}"

j11=$(submit_train "jcM2A11" "RUN_NAME=${CORE11},RECO_MAX_GENERATED_TOKENS=60,LOSS_W_BUDGET=0.75,LOSS_W_SPARSE=0.009")
echo "  CORE11 ${j11} ${CORE11}"

j12=$(submit_train "jcM2A12" "RUN_NAME=${CORE12},LOSS_W_LOCAL=0.14,LOSS_GEN_LOCAL_RADIUS=0.06,LOSS_W_SPARSE=0.013")
echo "  CORE12 ${j12} ${CORE12}"

dep_all="afterok:${j01}:${j02}:${j03}:${j04}:${j05}:${j06}:${j07}:${j08}:${j09}:${j10}:${j11}:${j12}"

MODEL_01_SPEC="m2_base:stage2:${SAVE_DIR}/${CORE01}"
MODEL_02_SPEC="m2_consstrong:stage2:${SAVE_DIR}/${CORE02}"
MODEL_03_SPEC="m2_budgetlite:stage2:${SAVE_DIR}/${CORE03}"
MODEL_04_SPEC="m2_genlow:stage2:${SAVE_DIR}/${CORE04}"
MODEL_05_SPEC="m2_genhigh:stage2:${SAVE_DIR}/${CORE05}"
MODEL_06_SPEC="m2_splitstrong:stage2:${SAVE_DIR}/${CORE06}"
MODEL_07_SPEC="m2_splitlight:stage2:${SAVE_DIR}/${CORE07}"
MODEL_08_SPEC="m2_reassignstrong:stage2:${SAVE_DIR}/${CORE08}"
MODEL_09_SPEC="m2_offdropmid:stage2:${SAVE_DIR}/${CORE09}"
MODEL_10_SPEC="m2_offdrophigh:stage2:${SAVE_DIR}/${CORE10}"
MODEL_11_SPEC="m2_topk60ish:stage2:${SAVE_DIR}/${CORE11}"
MODEL_12_SPEC="m2_antioverlap:stage2:${SAVE_DIR}/${CORE12}"

jfuse=$(submit_fusion "${dep_all}" \
  "SAVE_ROOT=${SAVE_DIR},OUT_DIR=${OUT_DIR},OPTIMIZE_FOR=${OPTIMIZE_FOR},MODEL_01_SPEC=${MODEL_01_SPEC},MODEL_02_SPEC=${MODEL_02_SPEC},MODEL_03_SPEC=${MODEL_03_SPEC},MODEL_04_SPEC=${MODEL_04_SPEC},MODEL_05_SPEC=${MODEL_05_SPEC},MODEL_06_SPEC=${MODEL_06_SPEC},MODEL_07_SPEC=${MODEL_07_SPEC},MODEL_08_SPEC=${MODEL_08_SPEC},MODEL_09_SPEC=${MODEL_09_SPEC},MODEL_10_SPEC=${MODEL_10_SPEC},MODEL_11_SPEC=${MODEL_11_SPEC},MODEL_12_SPEC=${MODEL_12_SPEC}")

echo "  FUSION ${jfuse} twelve-model stacked metafuser"

echo "============================================================"
echo "Queued JetClass 12-runner m2-hybrid 1M/250k/1M pipeline"
echo "Partition:   ${PARTITION}"
echo "Time limit:  ${TIME_LIMIT}"
echo "Train mem:   ${TRAIN_MEM:-runner-default}"
echo "Fusion mem:  ${FUSION_MEM:-runner-default}"
echo "Fusion job:  ${jfuse}"
echo "Fusion out:  ${OUT_DIR}"
echo "Dependency:  ${dep_all}"
echo "============================================================"
