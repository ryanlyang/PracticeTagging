#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

jid1=$(sbatch run_m25_k1_nomicro_bs80_smoke_50k20k100k.sh | awk '{print $4}')
echo "submitted m25 k1 quick no-micro: ${jid1}"

jid2=$(sbatch run_m25_k6_nomicro_bs64_smoke_50k20k100k.sh | awk '{print $4}')
echo "submitted m25 k6 quick no-micro: ${jid2}"

echo "done"
