#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

sbatch "sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m38_k6_seeded_m28_detresid_multicand_150k75k300k.sh"
