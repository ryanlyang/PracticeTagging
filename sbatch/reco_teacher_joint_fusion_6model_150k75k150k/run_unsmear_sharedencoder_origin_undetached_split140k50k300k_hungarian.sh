#!/usr/bin/env bash
#SBATCH --job-name=unsm14050300h
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --output=offline_reconstructor_logs/unsmear_sharedencoder_joint_new/unsmear_sharedencoder_split140k50k300k_hungarian_%j.out
#SBATCH --error=offline_reconstructor_logs/unsmear_sharedencoder_joint_new/unsmear_sharedencoder_split140k50k300k_hungarian_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/unsmear_sharedencoder_joint_new

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local start="$1"
  local cur
  cur="$(cd "${start}" 2>/dev/null && pwd || true)"
  if [[ -z "${cur}" ]]; then
    return 1
  fi
  while [[ "${cur}" != "/" ]]; do
    if [[ -f "${cur}/sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_unsmear_sharedencoder_origin_undetached_split140k50k300k.sh" ]]; then
      echo "${cur}"
      return 0
    fi
    cur="$(dirname "${cur}")"
  done
  return 1
}

if [[ -z "${REPO_ROOT:-}" ]]; then
  for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(pwd)" \
    "${SCRIPT_DIR}" \
    "${HOME}/atlas/PracticeTagging"
  do
    if [[ -n "${candidate}" ]]; then
      if REPO_FOUND="$(find_repo_root "${candidate}")"; then
        REPO_ROOT="${REPO_FOUND}"
        break
      fi
    fi
  done
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "ERROR: Could not locate repo root for Hungarian unsmear wrapper." >&2
  exit 2
fi

BASE_SCRIPT="${REPO_ROOT}/sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_unsmear_sharedencoder_origin_undetached_split140k50k300k.sh"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: Base script not found: ${BASE_SCRIPT}" >&2
  exit 2
fi

export JOINT_UNSMEAR_LOSS_MODE="${JOINT_UNSMEAR_LOSS_MODE:-hungarian}"
export RUN_NAME="${RUN_NAME:-unsmear_transformer_sharedencoder_delta_gate_joint_split140k50k300k_hungarian_seed42}"

echo "============================================================"
echo "Wrapper: Hungarian unsmear-loss ablation"
echo "Repo root: ${REPO_ROOT}"
echo "Base script: ${BASE_SCRIPT}"
echo "JOINT_UNSMEAR_LOSS_MODE=${JOINT_UNSMEAR_LOSS_MODE}"
echo "RUN_NAME=${RUN_NAME}"
echo "============================================================"

bash "${BASE_SCRIPT}"

