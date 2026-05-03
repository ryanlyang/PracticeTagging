#!/usr/bin/env bash
set -euo pipefail

# Wait for a set of SLURM jobs to finish.
# Usage:
#   bash sbatch/reco_teacher_joint_fusion_6model_150k75k150k/wait_m38_prefix6_jobs.sh
#   bash sbatch/reco_teacher_joint_fusion_6model_150k75k150k/wait_m38_prefix6_jobs.sh 123 456 789

POLL_SEC="${POLL_SEC:-30}"

if [[ "$#" -gt 0 ]]; then
  JOB_IDS=("$@")
else
  JOB_IDS=(21253956 21253957 21253958 21253959 21253960 21253961)
fi

echo "Waiting on jobs: ${JOB_IDS[*]}"
echo "Polling every ${POLL_SEC}s"

terminal_failed=()

while true; do
  pending_count=0
  terminal_failed=()

  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] status check"

  for jid in "${JOB_IDS[@]}"; do
    # First prefer live queue state.
    live_state="$(squeue -h -j "${jid}" -o '%T' | head -n1 || true)"
    if [[ -n "${live_state}" ]]; then
      state="${live_state}"
    else
      # Fall back to accounting once the job leaves queue.
      state="$(sacct -j "${jid}" -X -n -o State | head -n1 | awk '{print $1}' || true)"
    fi

    if [[ -z "${state}" ]]; then
      state="UNKNOWN"
    fi

    printf '  job %s -> %s\n' "${jid}" "${state}"

    case "${state}" in
      COMPLETED)
        ;;
      PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|STAGE_OUT)
        pending_count=$((pending_count + 1))
        ;;
      *)
        terminal_failed+=("${jid}:${state}")
        ;;
    esac
  done

  if [[ "${pending_count}" -eq 0 ]]; then
    break
  fi

  sleep "${POLL_SEC}"
done

if [[ "${#terminal_failed[@]}" -gt 0 ]]; then
  echo
  echo "One or more jobs ended in non-COMPLETED states:"
  printf '  %s\n' "${terminal_failed[@]}"
  exit 2
fi

echo
echo "All target jobs completed successfully."
