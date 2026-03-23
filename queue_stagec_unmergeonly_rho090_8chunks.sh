#!/usr/bin/env bash
# Queue all 8 chunked Stage-C sweep submitters (4 runs each).
set -euo pipefail

chunks=(
  submit_stagec_unmergeonly_rho090_chunk01.sh
  submit_stagec_unmergeonly_rho090_chunk02.sh
  submit_stagec_unmergeonly_rho090_chunk03.sh
  submit_stagec_unmergeonly_rho090_chunk04.sh
  submit_stagec_unmergeonly_rho090_chunk05.sh
  submit_stagec_unmergeonly_rho090_chunk06.sh
  submit_stagec_unmergeonly_rho090_chunk07.sh
  submit_stagec_unmergeonly_rho090_chunk08.sh
)

for c in "${chunks[@]}"; do
  if [[ ! -f "$c" ]]; then
    echo "Missing chunk script: $c" >&2
    exit 1
  fi
done

echo "Queueing all 8 chunk submitters..."
for c in "${chunks[@]}"; do
  echo "Launching $c"
  bash "$c"
done
echo "All chunk submitters completed."
