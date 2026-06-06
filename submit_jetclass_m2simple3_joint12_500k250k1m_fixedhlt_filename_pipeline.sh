#!/usr/bin/env bash
set -euo pipefail

# Queues 12 JetClass m2-simple3 hybrid adaptive-generation variants at 500k/250k/1M,
# then a dependent 12-model stacked metafuser (optimize_for=acc).
#
# This version keeps the HLT corruption profile fixed for every model and uses
# filename-derived JetClass labels to avoid the slow canonical label-branch scan.

PARTITION="${PARTITION:-tier3}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-${TIME_LIMIT:-2-00:00:00}}"
FUSION_TIME_LIMIT="${FUSION_TIME_LIMIT:-12:00:00}"
TRAIN_MEM="${TRAIN_MEM:-256G}"
FUSION_MEM="${FUSION_MEM:-128G}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"

RUNNER_TRAIN="${RUNNER_TRAIN:-run_train_jetclass_joint_dualview_v2attr_500k250k1m_m2hlt_simple3ops.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_twelve_model_stacked_fusion_500k250k1m_m2simple3.sh}"

for f in "${RUNNER_TRAIN}" "${RUNNER_FUSION}"; do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

TAG="${TAG:-fixedhlt}"
CORE01="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core01_base"
CORE02="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core02_consstrong"
CORE03="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core03_budgetlite"
CORE04="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core04_genlow"
CORE05="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core05_genhigh"
CORE06="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core06_splitstrong"
CORE07="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core07_splitlight"
CORE08="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core08_physstrong"
CORE09="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core09_offdropmid"
CORE10="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core10_offdrophigh"
CORE11="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core11_topk60ish"
CORE12="jetclass_joint_v2attr_500k250k1m_m2hlt_simple3ops_${TAG}_core12_antioverlap"

OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/twelve_model_500k250k1m_m2simple3_${TAG}_stacked_acc}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"

# One fixed train/val/test split, one fixed HLT simulation profile, one seed.
COMMON_EXPORT="SAVE_DIR=${SAVE_DIR},SEED=52,N_TRAIN_JETS=500000,N_VAL_JETS=250000,N_TEST_JETS=1000000,FEATURE_PREPROCESSING=canonical,CLASS_ASSIGNMENT=filename,TARGET_CLASS=Hbb,BACKGROUND_CLASS=QCD,HLT_PT_THRESHOLD=1.30,MERGE_PROB_SCALE=1.35,REASSIGN_SCALE=1.00,SMEAR_SCALE=1.00,EFF_PLATEAU_BARREL=0.99,EFF_PLATEAU_ENDCAP=0.97,EFF_TURNON_PT=1.40,EFF_WIDTH_PT=0.20"

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
    --time="${TRAIN_TIME_LIMIT}" \
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
    --job-name="jc12S3F" \
    --partition="${PARTITION}" \
    --time="${FUSION_TIME_LIMIT}" \
    "${mem_args[@]}" \
    --dependency="${dep}" \
    --export="ALL,${export_kv}" \
    "${RUNNER_FUSION}"
}

echo "Submitting fixed-HLT JetClass 12-runner m2-simple3 500k/250k/1M pipeline on ${PARTITION}"
echo "Train time: ${TRAIN_TIME_LIMIT}"
echo "Fusion time: ${FUSION_TIME_LIMIT}"
echo "Fixed HLT: pt_thr=1.30 merge=1.35 reassign=1.00 smear=1.00 eff=0.99/0.97 turnon=1.40 width=0.20"
echo "Labels:    filename"
echo "Seed:      52"

j01=$(submit_train "jcS301" "RUN_NAME=${CORE01}")
echo "  CORE01 ${j01} ${CORE01}"

j02=$(submit_train "jcS302" "RUN_NAME=${CORE02},LAMBDA_CONS=0.06,LOSS_W_BUDGET=0.65,STAGEC_LR_DUAL=1e-4,STAGEC_LR_RECO=5e-5")
echo "  CORE02 ${j02} ${CORE02}"

j03=$(submit_train "jcS303" "RUN_NAME=${CORE03},LOSS_W_BUDGET=0.25,LOSS_W_PT_RATIO=0.18,LOSS_W_SPARSE=0.012")
echo "  CORE03 ${j03} ${CORE03}"

j04=$(submit_train "jcS304" "RUN_NAME=${CORE04},RECO_MAX_GENERATED_TOKENS=40,LOSS_W_SPARSE=0.012")
echo "  CORE04 ${j04} ${CORE04}"

j05=$(submit_train "jcS305" "RUN_NAME=${CORE05},RECO_MAX_GENERATED_TOKENS=72,LOSS_W_SPARSE=0.006")
echo "  CORE05 ${j05} ${CORE05}"

j06=$(submit_train "jcS306" "RUN_NAME=${CORE06},LOSS_W_LOCAL=0.10,LOSS_W_BUDGET=0.55")
echo "  CORE06 ${j06} ${CORE06}"

j07=$(submit_train "jcS307" "RUN_NAME=${CORE07},LOSS_W_LOCAL=0.02,LOSS_W_BUDGET=0.30")
echo "  CORE07 ${j07} ${CORE07}"

j08=$(submit_train "jcS308" "RUN_NAME=${CORE08},LOSS_W_PT_RATIO=0.22,LOSS_W_M_RATIO=0.05,LOSS_W_E_RATIO=0.05,SIMPLE3_LOSS_W_AXIS=0.12")
echo "  CORE08 ${j08} ${CORE08}"

j09=$(submit_train "jcS309" "RUN_NAME=${CORE09},TARGET_DROP_PROB_MAX=0.50,TARGET_DROP_WARMUP_EPOCHS=20,TARGET_DROP_NUM_BANKS=3,TARGET_DROP_BANK_CYCLE_EPOCHS=1")
echo "  CORE09 ${j09} ${CORE09}"

j10=$(submit_train "jcS310" "RUN_NAME=${CORE10},TARGET_DROP_PROB_MAX=0.70,TARGET_DROP_WARMUP_EPOCHS=20,TARGET_DROP_NUM_BANKS=3,TARGET_DROP_BANK_CYCLE_EPOCHS=1")
echo "  CORE10 ${j10} ${CORE10}"

j11=$(submit_train "jcS311" "RUN_NAME=${CORE11},RECO_MAX_GENERATED_TOKENS=60,LOSS_W_BUDGET=0.50,LOSS_W_SPARSE=0.009")
echo "  CORE11 ${j11} ${CORE11}"

j12=$(submit_train "jcS312" "RUN_NAME=${CORE12},LOSS_W_LOCAL=0.14,LOSS_GEN_LOCAL_RADIUS=0.06,LOSS_W_SPARSE=0.013,SIMPLE3_LOSS_W_AXIS=0.14")
echo "  CORE12 ${j12} ${CORE12}"

dep_all="afterok:${j01}:${j02}:${j03}:${j04}:${j05}:${j06}:${j07}:${j08}:${j09}:${j10}:${j11}:${j12}"

MODEL_01_SPEC="m2_base:stage2:${SAVE_DIR}/${CORE01}"
MODEL_02_SPEC="m2_consstrong:stage2:${SAVE_DIR}/${CORE02}"
MODEL_03_SPEC="m2_budgetlite:stage2:${SAVE_DIR}/${CORE03}"
MODEL_04_SPEC="m2_genlow:stage2:${SAVE_DIR}/${CORE04}"
MODEL_05_SPEC="m2_genhigh:stage2:${SAVE_DIR}/${CORE05}"
MODEL_06_SPEC="m2_splitstrong:stage2:${SAVE_DIR}/${CORE06}"
MODEL_07_SPEC="m2_splitlight:stage2:${SAVE_DIR}/${CORE07}"
MODEL_08_SPEC="m2_physstrong:stage2:${SAVE_DIR}/${CORE08}"
MODEL_09_SPEC="m2_offdropmid:stage2:${SAVE_DIR}/${CORE09}"
MODEL_10_SPEC="m2_offdrophigh:stage2:${SAVE_DIR}/${CORE10}"
MODEL_11_SPEC="m2_topk60ish:stage2:${SAVE_DIR}/${CORE11}"
MODEL_12_SPEC="m2_antioverlap:stage2:${SAVE_DIR}/${CORE12}"

jfuse=$(submit_fusion "${dep_all}" \
  "SAVE_ROOT=${SAVE_DIR},OUT_DIR=${OUT_DIR},OPTIMIZE_FOR=${OPTIMIZE_FOR},MODEL_01_SPEC=${MODEL_01_SPEC},MODEL_02_SPEC=${MODEL_02_SPEC},MODEL_03_SPEC=${MODEL_03_SPEC},MODEL_04_SPEC=${MODEL_04_SPEC},MODEL_05_SPEC=${MODEL_05_SPEC},MODEL_06_SPEC=${MODEL_06_SPEC},MODEL_07_SPEC=${MODEL_07_SPEC},MODEL_08_SPEC=${MODEL_08_SPEC},MODEL_09_SPEC=${MODEL_09_SPEC},MODEL_10_SPEC=${MODEL_10_SPEC},MODEL_11_SPEC=${MODEL_11_SPEC},MODEL_12_SPEC=${MODEL_12_SPEC}")

echo "  FUSION ${jfuse} twelve-model stacked metafuser"

echo "============================================================"
echo "Queued fixed-HLT JetClass 12-runner m2-simple3 500k/250k/1M pipeline"
echo "Partition:   ${PARTITION}"
echo "Train time:  ${TRAIN_TIME_LIMIT}"
echo "Fusion time: ${FUSION_TIME_LIMIT}"
echo "Train mem:   ${TRAIN_MEM:-runner-default}"
echo "Fusion mem:  ${FUSION_MEM:-runner-default}"
echo "Fusion job:  ${jfuse}"
echo "Fusion out:  ${OUT_DIR}"
echo "Dependency:  ${dep_all}"
echo "============================================================"
