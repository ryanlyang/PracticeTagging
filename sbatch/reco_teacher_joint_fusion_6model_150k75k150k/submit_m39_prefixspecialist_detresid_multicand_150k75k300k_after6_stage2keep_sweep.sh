#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m39_prefixspecialist_detresid_multicand_150k75k300k.sh"
STAGE2_RUNNER="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m39_prefix6_stage2_150k75k300k.sh"

BASE_RUN_NAME="${BASE_RUN_NAME:-model39_prefixspecialist_detresid_multicand_150k75k300k_seed0}"
PREFIXES=(0 3 6 9 12 15)

SUBMIT_STAGE2="${SUBMIT_STAGE2:-1}"
STAGE2_KEEP_M_VALUES=(6 4 2 1)
STAGE2_RUN_PREFIX="${STAGE2_RUN_PREFIX:-model39_prefix6_stage2_keepm}"

declare -a JOB_IDS=()
for pfx in "${PREFIXES[@]}"; do
  out=$(sbatch \
    --job-name="m39sp${pfx}" \
    --output="offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefixspecialist_detresid_multicand_p${pfx}_%j.out" \
    --error="offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefixspecialist_detresid_multicand_p${pfx}_%j.err" \
    --export=ALL,RUN_NAME="${BASE_RUN_NAME}_pfx${pfx}",SEED_CANDIDATE_K=1,SEED_KEEP_M=1,SEED_MAX_PREFIX="${pfx}",TRAIN_SPECIALIST_PREFIX="${pfx}",CARRY_TARGET_MODE=fixed_k,CARRY_TARGET_K=-1,CARRY_TARGET_THRESH_GATE=0 \
    "${RUNNER}")
  jid=$(echo "${out}" | awk '{print $4}')
  JOB_IDS+=("${jid}")
  echo "Submitted m39 specialist p${pfx}: ${jid}"
done

if [[ "${SUBMIT_STAGE2}" == "1" ]]; then
  if [[ ! -f "${STAGE2_RUNNER}" ]]; then
    echo "Missing stage2 runner: ${STAGE2_RUNNER}" >&2
    exit 1
  fi

  dep="$(IFS=:; echo "${JOB_IDS[*]}")"
  echo "Stage2 dependency: afterok:${dep}"

  for keep_m in "${STAGE2_KEEP_M_VALUES[@]}"; do
    run_name="${STAGE2_RUN_PREFIX}${keep_m}_150k75k300k_seed0"
    out2=$(sbatch \
      --dependency="afterok:${dep}" \
      --job-name="m39s2k${keep_m}" \
      --output="offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefix6_stage2_k${keep_m}_%j.out" \
      --error="offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/m39_prefix6_stage2_k${keep_m}_%j.err" \
      --export=ALL,RUN_NAME="${run_name}",STAGE2_KEEP_M="${keep_m}" \
      "${STAGE2_RUNNER}")
    jid2=$(echo "${out2}" | awk '{print $4}')
    echo "Submitted dependent m39 stage2 keep_m=${keep_m}: ${jid2}"
  done
fi
