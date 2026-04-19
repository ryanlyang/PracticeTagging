#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/../.."

RUN_DIR="sbatch/reco_teacher_joint_fusion_6model_150k75k150k"

# Extra seeds to run beyond the existing seed0 runs.
SEEDS_STR="${SEEDS_STR:-1 2}"
read -r -a SEEDS <<< "${SEEDS_STR}"

RUN_SCRIPTS=(
  "run_teacher_hlt_only_1m250k250k.sh"
  "run_teacher_hlt_only_2m500k500k.sh"
  "run_teacher_hlt_only_3m750k750k.sh"
  "run_teacher_hlt_only_4m1m1m.sh"
  "run_teacher_hlt_only_5m1250k1250k.sh"
  "run_teacher_hlt_only_6m1500k1500k.sh"
  "run_teacher_hlt_only_7m1750k1750k.sh"
)

echo "Submitting teacher/HLT-only runs for seeds: ${SEEDS[*]}"
echo "Run set: 1M..7M"

for seed in "${SEEDS[@]}"; do
  for script in "${RUN_SCRIPTS[@]}"; do
    tag="${script#run_}"
    tag="${tag%.sh}"
    run_name="${tag}_seed${seed}"
    script_path="${RUN_DIR}/${script}"

    # Stability overrides for high-count jobs.
    mem_override=""
    workers_override=""
    if [[ "${tag}" == "teacher_hlt_only_6m1500k1500k" ]]; then
      mem_override="320G"
      workers_override="2"
    elif [[ "${tag}" == "teacher_hlt_only_7m1750k1750k" ]]; then
      mem_override="384G"
      workers_override="1"
    fi

    export_args="ALL,SEED=${seed},RUN_NAME=${run_name}"
    if [[ -n "${workers_override}" ]]; then
      export_args+=",NUM_WORKERS=${workers_override}"
    fi

    if [[ -n "${mem_override}" ]]; then
      jid="$(sbatch --mem="${mem_override}" --export="${export_args}" "${script_path}" | awk '{print $4}')"
      echo "seed=${seed} ${tag} -> job=${jid} (mem=${mem_override}, workers=${workers_override:-default})"
    else
      jid="$(sbatch --export="${export_args}" "${script_path}" | awk '{print $4}')"
      echo "seed=${seed} ${tag} -> job=${jid}"
    fi
  done
done

echo "Submitted all jobs."
echo "Monitor: squeue -u ${USER} | rg 'th|teacher_hlt_only'"
