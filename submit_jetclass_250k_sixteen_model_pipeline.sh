#!/usr/bin/env bash
set -euo pipefail

# Queues a 16-model JetClass 250k/50k/250k pipeline:
# - 13 core variants (mostly standalone, analyze12-like behavior)
# - 3 explicitly requested runs:
#   1) ..._path_gentok56_ablate_lcons003_recoonlydual
#   2) ..._stagea_teacherlogit_autoteacher_recoonlydual
#   3) ..._offlineteacher_stagea_recoonlydual
# Then submits one dependent stacked-fusion run over all 16 outputs.

PARTITION="${PARTITION:-tier3}"
TIME_LIMIT="${TIME_LIMIT:-5-00:00:00}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"

RUNNER_V1="${RUNNER_V1:-run_train_jetclass_joint_dualview_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25.sh}"
RUNNER_PATH="${RUNNER_PATH:-run_train_jetclass_joint_dualview_confgen_v2attr_250k50k250k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual.sh}"
RUNNER_AUTOTEACH="${RUNNER_AUTOTEACH:-run_train_jetclass_joint_dualview_confgen_v2attr_250k50k250k_v1hltplus25_stagea_teacherlogit_autoteacher_recoonlydual.sh}"
RUNNER_OFFTEACH="${RUNNER_OFFTEACH:-run_train_jetclass_joint_dualview_confgen_v2attr_250k50k250k_v1hltplus25_offlineteacher_stagea_recoonlydual.sh}"
RUNNER_FUSION16="${RUNNER_FUSION16:-run_analyze_jetclass_sixteen_model_stacked_fusion_250k.sh}"

for f in \
  "${RUNNER_V1}" "${RUNNER_PATH}" \
  "${RUNNER_AUTOTEACH}" "${RUNNER_OFFTEACH}" "${RUNNER_FUSION16}"
do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

CORE01="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core01_base"
CORE02="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core02_joint"
CORE03="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core03_genlow"
CORE04="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core04_genhigh"
CORE05="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core05_splitstrong"
CORE06="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core06_splitlight"
CORE07="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core07_path_prejoint"
CORE08="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core08_path_joint"
CORE09="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core09_path_sparsegen_low"
CORE10="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core10_path_sparsegen_high"
CORE11="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core11_autoteach_base"
CORE12="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core12_autoteach_teacherdom"
CORE13="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_core13_autoteach_setheavy"

# Explicitly requested three runs.
EXP14="jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual"
EXP15="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_stagea_teacherlogit_autoteacher_recoonlydual"
EXP16="jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_offlineteacher_stagea_recoonlydual"

OUT_DIR="${OUT_DIR:-${SAVE_DIR}/fusion_reports/sixteen_model_250k_stacked_acc}"
OPTIMIZE_FOR="${OPTIMIZE_FOR:-acc}"

submit_job() {
  local runner="$1"
  local export_kv="$2"
  local dep="${3:-}"
  if [[ -n "${dep}" ]]; then
    sbatch --parsable \
      --partition="${PARTITION}" \
      --time="${TIME_LIMIT}" \
      --dependency="${dep}" \
      --export="ALL,${export_kv}" \
      "${runner}"
  else
    sbatch --parsable \
      --partition="${PARTITION}" \
      --time="${TIME_LIMIT}" \
      --export="ALL,${export_kv}" \
      "${runner}"
  fi
}

echo "Submitting 16-model JetClass pipeline on ${PARTITION} with time limit ${TIME_LIMIT}"

# Core 1-6: v1 HLT family.
j01=$(submit_job "${RUNNER_V1}" "RUN_NAME=${CORE01},STAGEC_EPOCHS=0")
echo "  CORE01 ${j01} ${CORE01}"
j02=$(submit_job "${RUNNER_V1}" "RUN_NAME=${CORE02},STAGEC_EPOCHS=45")
echo "  CORE02 ${j02} ${CORE02}"
j03=$(submit_job "${RUNNER_V1}" "RUN_NAME=${CORE03},STAGEC_EPOCHS=0,JETCLASS_STAGEA_W_SPARSE_GEN=0.0015,JETCLASS_STAGEA_W_GEN_FP=0.02")
echo "  CORE03 ${j03} ${CORE03}"
j04=$(submit_job "${RUNNER_V1}" "RUN_NAME=${CORE04},STAGEC_EPOCHS=0,JETCLASS_STAGEA_W_SPARSE_GEN=0.0060,JETCLASS_STAGEA_W_GEN_FP=0.08")
echo "  CORE04 ${j04} ${CORE04}"
j05=$(submit_job "${RUNNER_V1}" "RUN_NAME=${CORE05},STAGEC_EPOCHS=0,JETCLASS_STAGEA_W_SPARSE_SPLIT=0.016")
echo "  CORE05 ${j05} ${CORE05}"
j06=$(submit_job "${RUNNER_V1}" "RUN_NAME=${CORE06},STAGEC_EPOCHS=0,JETCLASS_STAGEA_W_SPARSE_SPLIT=0.008")
echo "  CORE06 ${j06} ${CORE06}"

# Core 7-10: path-family standalone.
j07=$(submit_job "${RUNNER_PATH}" "RUN_NAME=${CORE07},STAGEC_EPOCHS=0")
echo "  CORE07 ${j07} ${CORE07}"
j08=$(submit_job "${RUNNER_PATH}" "RUN_NAME=${CORE08},STAGEC_EPOCHS=45")
echo "  CORE08 ${j08} ${CORE08}"
j09=$(submit_job "${RUNNER_PATH}" "RUN_NAME=${CORE09},JETCLASS_STAGEA_W_SPARSE_GEN=0.0015,JETCLASS_STAGEA_W_GEN_FP=0.02")
echo "  CORE09 ${j09} ${CORE09}"
j10=$(submit_job "${RUNNER_PATH}" "RUN_NAME=${CORE10},JETCLASS_STAGEA_W_SPARSE_GEN=0.0060,JETCLASS_STAGEA_W_GEN_FP=0.08")
echo "  CORE10 ${j10} ${CORE10}"

# Core 11-13: autoteacher-family standalone.
j11=$(submit_job "${RUNNER_AUTOTEACH}" "RUN_NAME=${CORE11},STAGEC_EPOCHS=0")
echo "  CORE11 ${j11} ${CORE11}"
j12=$(submit_job "${RUNNER_AUTOTEACH}" "RUN_NAME=${CORE12},STAGEC_EPOCHS=0,JETCLASS_STAGEA_LAMBDA_TEACHER=1.30,JETCLASS_STAGEA_LAMBDA_BUDGET_ONLY=0.25,JETCLASS_STAGEA_LAMBDA_SET_ONLY=0.01,JETCLASS_STAGEA_LAMBDA_LOCAL_ONLY=0.005")
echo "  CORE12 ${j12} ${CORE12}"
j13=$(submit_job "${RUNNER_AUTOTEACH}" "RUN_NAME=${CORE13},STAGEC_EPOCHS=0,JETCLASS_STAGEA_LAMBDA_TEACHER=0.80,JETCLASS_STAGEA_LAMBDA_BUDGET_ONLY=0.35,JETCLASS_STAGEA_LAMBDA_SET_ONLY=0.05,JETCLASS_STAGEA_LAMBDA_LOCAL_ONLY=0.02")
echo "  CORE13 ${j13} ${CORE13}"

# Explicit 14: requested path run.
j14=$(submit_job "${RUNNER_PATH}" "RUN_NAME=${EXP14}")
echo "  EXP14  ${j14} ${EXP14}"

# Explicit 15: auto-teacher run.
j15=$(submit_job "${RUNNER_AUTOTEACH}" "RUN_NAME=${EXP15}")
echo "  EXP15  ${j15} ${EXP15}"

# Explicit 16: offlineteacher run; default uses EXP15 teacher checkpoint.
# Set EXP16_TEACHER_CKPT to decouple it from EXP15 if you have a fixed checkpoint.
EXP16_TEACHER_CKPT="${EXP16_TEACHER_CKPT:-${SAVE_DIR}/${EXP15}_teacherkin/teacher_offline_best.pt}"
EXP16_DEP="${EXP16_DEP:-afterok:${j15}}"
j16=$(submit_job "${RUNNER_OFFTEACH}" "RUN_NAME=${EXP16},TEACHER_CKPT=${EXP16_TEACHER_CKPT}" "${EXP16_DEP}")
echo "  EXP16  ${j16} ${EXP16}"

# Final fusion depends on all 16.
dep_all="afterok:${j01}:${j02}:${j03}:${j04}:${j05}:${j06}:${j07}:${j08}:${j09}:${j10}:${j11}:${j12}:${j13}:${j14}:${j15}:${j16}"
MODEL_01_SPEC="v1_base:stage2:${SAVE_DIR}/${CORE01}"
MODEL_02_SPEC="v1_joint:stage2:${SAVE_DIR}/${CORE02}"
MODEL_03_SPEC="v1_genlow:stage2:${SAVE_DIR}/${CORE03}"
MODEL_04_SPEC="v1_genhigh:stage2:${SAVE_DIR}/${CORE04}"
MODEL_05_SPEC="v1_splitstrong:stage2:${SAVE_DIR}/${CORE05}"
MODEL_06_SPEC="v1_splitlight:stage2:${SAVE_DIR}/${CORE06}"
MODEL_07_SPEC="path_prejoint:stage2:${SAVE_DIR}/${CORE07}"
MODEL_08_SPEC="path_joint:stage2:${SAVE_DIR}/${CORE08}"
MODEL_09_SPEC="path_sparsegen_low:stage2:${SAVE_DIR}/${CORE09}"
MODEL_10_SPEC="path_sparsegen_high:stage2:${SAVE_DIR}/${CORE10}"
MODEL_11_SPEC="autoteach_base:stage2:${SAVE_DIR}/${CORE11}"
MODEL_12_SPEC="autoteach_teacherdom:stage2:${SAVE_DIR}/${CORE12}"
MODEL_13_SPEC="autoteach_setheavy:stage2:${SAVE_DIR}/${CORE13}"
MODEL_14_SPEC="path_lcons003_explicit:stage2:${SAVE_DIR}/${EXP14}"
MODEL_15_SPEC="autoteacher_explicit:stage2:${SAVE_DIR}/${EXP15}"
MODEL_16_SPEC="offlineteacher_explicit:stage2:${SAVE_DIR}/${EXP16}"

jfuse=$(submit_job "${RUNNER_FUSION16}" \
  "OUT_DIR=${OUT_DIR},OPTIMIZE_FOR=${OPTIMIZE_FOR},MODEL_01_SPEC=${MODEL_01_SPEC},MODEL_02_SPEC=${MODEL_02_SPEC},MODEL_03_SPEC=${MODEL_03_SPEC},MODEL_04_SPEC=${MODEL_04_SPEC},MODEL_05_SPEC=${MODEL_05_SPEC},MODEL_06_SPEC=${MODEL_06_SPEC},MODEL_07_SPEC=${MODEL_07_SPEC},MODEL_08_SPEC=${MODEL_08_SPEC},MODEL_09_SPEC=${MODEL_09_SPEC},MODEL_10_SPEC=${MODEL_10_SPEC},MODEL_11_SPEC=${MODEL_11_SPEC},MODEL_12_SPEC=${MODEL_12_SPEC},MODEL_13_SPEC=${MODEL_13_SPEC},MODEL_14_SPEC=${MODEL_14_SPEC},MODEL_15_SPEC=${MODEL_15_SPEC},MODEL_16_SPEC=${MODEL_16_SPEC}" \
  "${dep_all}")
echo "  FUSION ${jfuse} sixteen-model stacked fusion"

echo "============================================================"
echo "Queued JetClass 16-model pipeline"
echo "Partition:   ${PARTITION}"
echo "Time limit:  ${TIME_LIMIT}"
echo "Fusion job:  ${jfuse}"
echo "Fusion out:  ${OUT_DIR}"
echo "Dependency:  ${dep_all}"
echo "============================================================"
