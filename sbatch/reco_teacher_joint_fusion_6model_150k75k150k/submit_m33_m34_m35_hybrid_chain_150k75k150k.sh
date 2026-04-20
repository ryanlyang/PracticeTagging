#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

M33_RUNNER="run_m33_k6_detfeas_dualview_postrefine_150k75k150k.sh"
M34_RUNNER="run_m34_k12_globalcand_multiview3_150k75k150k.sh"
M35_RUNNER="run_m35_hybrid_m33m34_150k75k150k.sh"

jid33=$(sbatch "$M33_RUNNER" | awk '{print $4}')
jid34=$(sbatch "$M34_RUNNER" | awk '{print $4}')
jid35=$(sbatch --dependency=afterok:${jid33}:${jid34} "$M35_RUNNER" | awk '{print $4}')

echo "Submitted chain:"
echo "  m33 postrefine jobid: $jid33"
echo "  m34 postrefine jobid: $jid34"
echo "  m35 hybrid jobid:     $jid35 (afterok:$jid33:$jid34)"
