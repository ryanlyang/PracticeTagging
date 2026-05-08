#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
# 1) Build fused targets from two completed teacher runs.
# 2) Train offline fused student on those targets.
# 3) Train confgen reconstructor/dualview with Stage-A teacher-dominant loss.

RUNNER_BUILD="${RUNNER_BUILD:-run_build_jetclass_fused_targets_v1hltplus25_pair.sh}"
RUNNER_STUDENT="${RUNNER_STUDENT:-run_train_jetclass_offline_fused_student_v1hltplus25_pair.sh}"
RUNNER_FINAL="${RUNNER_FINAL:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_fusedstudent_stagea_teacher_recoonlydual.sh}"

RUN_A_DIR="${RUN_A_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
RUN_B_DIR="${RUN_B_DIR:-checkpoints/jetclass_joint_dualview/jetclass_joint_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25_gentok56}"
TARGETS_DIR="${TARGETS_DIR:-checkpoints/jetclass_joint_dualview/fused_targets/$(basename "${RUN_A_DIR}")__AND__$(basename "${RUN_B_DIR}")}"
STUDENT_RUN_NAME="${STUDENT_RUN_NAME:-jetclass_offline_fused_student_50k25k100k_v1hltplus25_pair}"
STUDENT_CKPT="${STUDENT_CKPT:-checkpoints/jetclass_joint_dualview/${STUDENT_RUN_NAME}/offline_fused_student.pt}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_v1hltplus25_fusedstudent_stagea_teacher_recoonlydual}"
UPSTREAM_TEACHER_JOBIDS="${UPSTREAM_TEACHER_JOBIDS:-}"

for f in "${RUNNER_BUILD}" "${RUNNER_STUDENT}" "${RUNNER_FINAL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing runner: ${f}" >&2
    exit 1
  fi
done

echo "Submitting fused-target build job..."
if [[ -n "${UPSTREAM_TEACHER_JOBIDS}" ]]; then
  echo "  Upstream teacher dependency: afterok:${UPSTREAM_TEACHER_JOBIDS}"
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

echo "Submitting offline fused-student job (afterok:${job_build})..."
job_student=$(sbatch --parsable \
  --dependency=afterok:${job_build} \
  --export=ALL,TARGETS_DIR="${TARGETS_DIR}",RUN_NAME="${STUDENT_RUN_NAME}" \
  "${RUNNER_STUDENT}")
echo "  Student job ID: ${job_student}"

echo "Submitting final Stage-A teacher run (afterok:${job_student})..."
job_final=$(sbatch --parsable \
  --dependency=afterok:${job_student} \
  --export=ALL,FUSED_STUDENT_CKPT="${STUDENT_CKPT}",RUN_NAME="${FINAL_RUN_NAME}" \
  "${RUNNER_FINAL}")
echo "  Final job ID: ${job_final}"

echo "============================================================"
echo "Queued fused-student pipeline"
echo "Build job:   ${job_build}"
echo "Student job: ${job_student}"
echo "Final job:   ${job_final}"
echo "Targets dir: ${TARGETS_DIR}"
echo "Student ckpt:${STUDENT_CKPT}"
echo "Final run:   ${FINAL_RUN_NAME}"
echo "============================================================"
