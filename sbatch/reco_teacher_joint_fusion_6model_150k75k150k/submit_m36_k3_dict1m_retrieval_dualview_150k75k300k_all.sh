#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sbatch run_m36_k3_dict1m_retrieval_dualview_150k75k300k.sh
