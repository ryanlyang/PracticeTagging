#!/usr/bin/env bash
# Queue all 8 chunked Stage-C jobs (each chunk runs 4 configs sequentially).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

mkdir -p offline_reconstructor_logs
stamp="$(date +%Y%m%d_%H%M%S)"
manifest="offline_reconstructor_logs/stagec_unmergeonly_rho090_queue_${stamp}.tsv"
printf "chunk_script\tjob_id\n" > "$manifest"

echo "Queueing all 8 chunk jobs (each runs 4 configs sequentially)..."
for c in "${chunks[@]}"; do
  out="$(sbatch --chdir "$SCRIPT_DIR" "$c")"
  job_id="${out##* }"
  printf "%s\t%s\n" "$c" "$job_id" >> "$manifest"
  echo "$out | $c"
done

echo "Done. Queue manifest: $manifest"
