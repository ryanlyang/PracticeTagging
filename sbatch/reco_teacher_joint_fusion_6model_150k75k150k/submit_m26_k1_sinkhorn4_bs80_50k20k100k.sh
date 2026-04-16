#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
jid=$(sbatch run_m26_k1_sinkhorn4_bs80_50k20k100k.sh | awk '{print $4}')
echo "submitted m26 k1 sinkhorn4 bs80: ${jid}"
