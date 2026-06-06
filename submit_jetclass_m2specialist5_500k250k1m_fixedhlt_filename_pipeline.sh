#!/usr/bin/env bash
set -euo pipefail

# Queues five fixed-HLT JetClass reconstruction specialists plus a dependent
# stacked-logistic fusion job. Every specialist uses the same data split, seed,
# and m2 HLT generation; only reconstructor behavior/losses change.

PARTITION="${PARTITION:-tier3}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-2-00:00:00}"
FUSION_TIME_LIMIT="${FUSION_TIME_LIMIT:-1-00:00:00}"
TRAIN_MEM="${TRAIN_MEM:-224G}"
FUSION_MEM="${FUSION_MEM:-128G}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
TAG="${TAG:-fixedhlt}"

RUNNER_TRAIN="${RUNNER_TRAIN:-run_train_jetclass_joint_dualview_v2attr_500k250k1m_m2hlt_specialist5.sh}"
RUNNER_FUSION="${RUNNER_FUSION:-run_analyze_jetclass_five_model_stacked_fusion_500k250k1m_m2specialist5.sh}"

for f in "${RUNNER_TRAIN}" "${RUNNER_FUSION}"; do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

GENERALIST="jetclass_joint_v2attr_500k250k1m_m2specialist5_${TAG}_generalist"
GLOBAL="jetclass_joint_v2attr_500k250k1m_m2specialist5_${TAG}_global_kinematic"
LOW_SPLIT="jetclass_joint_v2attr_500k250k1m_m2specialist5_${TAG}_low_split"
LOW_GEN="jetclass_joint_v2attr_500k250k1m_m2specialist5_${TAG}_low_generate"
LOW_EDIT="jetclass_joint_v2attr_500k250k1m_m2specialist5_${TAG}_low_edit"
OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/five_model_500k250k1m_m2specialist5_${TAG}_stacked_acc}"
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
    --job-name="jc5SpF" \
    --partition="${PARTITION}" \
    --time="${FUSION_TIME_LIMIT}" \
    --mem="${FUSION_MEM}" \
    --dependency="${dep}" \
    --export="ALL,${export_kv}" \
    "${RUNNER_FUSION}"
}

echo "Submitting JetClass five-specialist fixed-HLT pipeline"
echo "Split: train=500k val=250k test=1m"
echo "Fixed HLT: pt_thr=1.30 merge=1.35 reassign=1.00 smear=1.00 eff=0.99/0.97 turnon=1.40 width=0.20"
echo "Train time/mem: ${TRAIN_TIME_LIMIT} / ${TRAIN_MEM}"
echo "Fusion time/mem: ${FUSION_TIME_LIMIT} / ${FUSION_MEM}"

j1=$(submit_train "jc5Sp01" "RUN_NAME=${GENERALIST}")
echo "  generalist       ${j1} ${GENERALIST}"

j2=$(submit_train "jc5Sp02" "RUN_NAME=${GLOBAL},LOSS_W_BUDGET=0.55,LOSS_W_PT_RATIO=0.35,LOSS_W_E_RATIO=0.10,LOSS_W_M_RATIO=0.08,LOSS_W_LOCAL=0.05,SPECIALIST_LOSS_W_AXIS=0.18,SPECIALIST_LOSS_W_RADIAL_PROFILE=0.03")
echo "  global_kinematic ${j2} ${GLOBAL}"

j3=$(submit_train "jc5Sp03" "RUN_NAME=${LOW_SPLIT},LOSS_W_BUDGET=0.65,LOSS_W_LOCAL=0.06,SPECIALIST_SPLIT_WEIGHT_SCALE=0.25,SPECIALIST_LOSS_W_SPLIT_SPARSE=0.08")
echo "  low_split        ${j3} ${LOW_SPLIT}"

j4=$(submit_train "jc5Sp04" "RUN_NAME=${LOW_GEN},RECO_MAX_GENERATED_TOKENS=24,LOSS_W_BUDGET=0.60,LOSS_W_LOCAL=0.08,SPECIALIST_GEN_WEIGHT_SCALE=0.25,SPECIALIST_LOSS_W_GEN_SPARSE=0.08")
echo "  low_generate     ${j4} ${LOW_GEN}"

j5=$(submit_train "jc5Sp05" "RUN_NAME=${LOW_EDIT},RECO_MAX_GENERATED_TOKENS=60,LOSS_W_BUDGET=0.75,LOSS_W_LOCAL=0.10,SPECIALIST_EDIT_DELTA_SCALE=0.25")
echo "  low_edit         ${j5} ${LOW_EDIT}"

dep_all="afterok:${j1}:${j2}:${j3}:${j4}:${j5}"

MODEL_01_SPEC="generalist:stage2:${SAVE_DIR}/${GENERALIST}"
MODEL_02_SPEC="global_kinematic:stage2:${SAVE_DIR}/${GLOBAL}"
MODEL_03_SPEC="low_split:stage2:${SAVE_DIR}/${LOW_SPLIT}"
MODEL_04_SPEC="low_generate:stage2:${SAVE_DIR}/${LOW_GEN}"
MODEL_05_SPEC="low_edit:stage2:${SAVE_DIR}/${LOW_EDIT}"

jfuse=$(submit_fusion "${dep_all}" \
  "SAVE_ROOT=${SAVE_DIR},TAG=${TAG},OUT_DIR=${OUT_DIR},OPTIMIZE_FOR=${OPTIMIZE_FOR},MODEL_01_SPEC=${MODEL_01_SPEC},MODEL_02_SPEC=${MODEL_02_SPEC},MODEL_03_SPEC=${MODEL_03_SPEC},MODEL_04_SPEC=${MODEL_04_SPEC},MODEL_05_SPEC=${MODEL_05_SPEC}")

echo "  fusion           ${jfuse} five-specialist stacked metafuser"

echo "============================================================"
echo "Queued JetClass five-specialist fixed-HLT pipeline"
echo "Fusion job:  ${jfuse}"
echo "Fusion out:  ${OUT_DIR}"
echo "Dependency:  ${dep_all}"
echo "============================================================"
