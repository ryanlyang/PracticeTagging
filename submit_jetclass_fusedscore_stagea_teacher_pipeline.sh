#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
# 1) (Optionally wait for two upstream teacher jobs.)
# 2) Build fused targets from run A/run B.
# 3) Train confgen run with Stage-A objective: teacher(reco) -> fused target.

RUNNER_BUILD="${RUNNER_BUILD:-run_build_jetclass_fused_targets_v1hltplus25_pair.sh}"
RUNNER_FINAL="${RUNNER_FINAL:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_fusedscore_stagea_teacher_recoonlydual.sh}"

RUN_A_DIR="${RUN_A_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
RUN_B_DIR="${RUN_B_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25_gentok56}"
TARGETS_DIR="${TARGETS_DIR:-checkpoints/jetclass_joint_dualview/fused_targets/$(basename "${RUN_A_DIR}")__AND__$(basename "${RUN_B_DIR}")}"
FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ:-${TARGETS_DIR}/fused_targets_train_val_test.npz}"
OFFLINE_TEACHER_CKPT="${OFFLINE_TEACHER_CKPT:-${RUN_B_DIR}/teacher.pt}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_v1hltplus25_fusedscore_stagea_teacher_recoonlydual}"
UPSTREAM_TEACHER_JOBIDS="${UPSTREAM_TEACHER_JOBIDS:-}"

for f in "${RUNNER_BUILD}" "${RUNNER_FINAL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing runner: ${f}" >&2
    exit 1
  fi
done

echo "Submitting fused-target build job..."
if [[ -n "${UPSTREAM_TEACHER_JOBIDS}" ]]; then
  echo "  Upstream dependency: afterok:${UPSTREAM_TEACHER_JOBIDS}"
  job_build=$(sbatch --parsable \
    --dependency=afterok:${UPSTREAM_TEACHER_JOBIDS} \
    --export=ALL,RUN_A_DIR="${RUN_A_DIR}",RUN_B_DIR="${RUN_B_DIR}",OUT_DIR="${TARGETS_DIR}" \
    "${RUNNER_BUILD}")
else
  job_build=$(sbatch --parsable \
    --export=ALL,RUN_A_DIR="${RUN_A_DIR}",RUN_B_DIR="${RUN_B_DIR}",OUT_DIR="${TARGETS_DIR}" \
    "${RUNNER_BUILD}")
fi
echo "  Build job ID: ${job_build}"

echo "Submitting final Stage-A fused-score teacher run (afterok:${job_build})..."
job_final=$(sbatch --parsable \
  --dependency=afterok:${job_build} \
  --export=ALL,RUN_A_DIR="${RUN_A_DIR}",RUN_B_DIR="${RUN_B_DIR}",FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ}",OFFLINE_TEACHER_CKPT="${OFFLINE_TEACHER_CKPT}",RUN_NAME="${FINAL_RUN_NAME}" \
  "${RUNNER_FINAL}")
echo "  Final job ID: ${job_final}"

echo "============================================================"
echo "Queued fused-score Stage-A teacher pipeline"
echo "Build job:    ${job_build}"
echo "Final job:    ${job_final}"
echo "Run A dir:    ${RUN_A_DIR}"
echo "Run B dir:    ${RUN_B_DIR}"
echo "Targets npz:  ${FUSED_TARGETS_NPZ}"
echo "Teacher ckpt: ${OFFLINE_TEACHER_CKPT}"
echo "Final run:    ${FINAL_RUN_NAME}"
echo "============================================================"

