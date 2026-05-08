#!/usr/bin/env bash
set -euo pipefail

# Queue plan (all on tier3):
# 1) Train Teacher A (confgen path variant).
# 2) Train Teacher B (v1hlt +25 profile).
# 3) Build fused bin-gated targets from A/B.
# 4) Strategy-2 prep: train offline fused student from those targets.
# 5) Strategy-1: direct two-teacher KD run.
# 6) Strategy-2: Stage-A teacher run from fused-student ckpt.
# 7) Strategy-3: Stage-A fusedscore run (teacher(reco) -> fused targets).
# 8) Extra requested run: plain Stage-A offline-teacher objective.

PARTITION="${PARTITION:-tier3}"

RUNNER_A="${RUNNER_A:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual.sh}"
RUNNER_B="${RUNNER_B:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_stronger_canonical_v1hlt_hltplus25.sh}"
RUNNER_BUILD="${RUNNER_BUILD:-run_build_jetclass_fused_targets_v1hltplus25_pair.sh}"
RUNNER_STUDENT="${RUNNER_STUDENT:-run_train_jetclass_offline_fused_student_v1hltplus25_pair.sh}"
RUNNER_FUSEDKD="${RUNNER_FUSEDKD:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_fusedkd_two_teacher.sh}"
RUNNER_FUSEDSTAGEA="${RUNNER_FUSEDSTAGEA:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_fusedstudent_stagea_teacher_recoonlydual.sh}"
RUNNER_FUSEDSCORE="${RUNNER_FUSEDSCORE:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_fusedscore_stagea_teacher_recoonlydual.sh}"
RUNNER_OFFTEACH="${RUNNER_OFFTEACH:-run_train_jetclass_joint_dualview_confgen_v2attr_50k25k100k_v1hltplus25_offlineteacher_stagea_recoonlydual.sh}"

SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
RUN_A_NAME="${RUN_A_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_tier3_teacherA_lcons003_recoonlydual}"
RUN_B_NAME="${RUN_B_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_tier3_teacherB_v1hltplus25_gentok56}"
TARGETS_DIR="${TARGETS_DIR:-${SAVE_DIR}/fused_targets/${RUN_A_NAME}__AND__${RUN_B_NAME}}"
STUDENT_RUN_NAME="${STUDENT_RUN_NAME:-jetclass_offline_fused_student_50k25k100k_tier3_${RUN_A_NAME}__${RUN_B_NAME}}"
STUDENT_CKPT="${STUDENT_CKPT:-${SAVE_DIR}/${STUDENT_RUN_NAME}/offline_fused_student.pt}"

RUN_FUSEDKD_NAME="${RUN_FUSEDKD_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_tier3_fusedkd_from_${RUN_A_NAME}_and_${RUN_B_NAME}}"
RUN_FUSEDSTAGEA_NAME="${RUN_FUSEDSTAGEA_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_tier3_fusedstudent_stagea_from_${RUN_A_NAME}_and_${RUN_B_NAME}}"
RUN_FUSEDSCORE_NAME="${RUN_FUSEDSCORE_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_tier3_fusedscore_stagea_from_${RUN_A_NAME}_and_${RUN_B_NAME}}"
RUN_OFFTEACH_NAME="${RUN_OFFTEACH_NAME:-jetclass_joint_confgen_v2attr_50k25k100k_tier3_offlineteacher_stagea_from_${RUN_B_NAME}}"

for f in \
  "${RUNNER_A}" "${RUNNER_B}" "${RUNNER_BUILD}" "${RUNNER_STUDENT}" \
  "${RUNNER_FUSEDKD}" "${RUNNER_FUSEDSTAGEA}" "${RUNNER_FUSEDSCORE}" "${RUNNER_OFFTEACH}"
do
  [[ -f "${f}" ]] || { echo "Missing runner: ${f}" >&2; exit 1; }
done

RUN_A_DIR="${SAVE_DIR}/${RUN_A_NAME}"
RUN_B_DIR="${SAVE_DIR}/${RUN_B_NAME}"
FUSED_TARGETS_NPZ="${TARGETS_DIR}/fused_targets_train_val_test.npz"

echo "Submitting Teacher A on ${PARTITION}..."
job_a=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,RUN_NAME="${RUN_A_NAME}" "${RUNNER_A}")
echo "  Teacher A job: ${job_a}"

echo "Submitting Teacher B on ${PARTITION}..."
job_b=$(sbatch --parsable --partition="${PARTITION}" --export=ALL,RUN_NAME="${RUN_B_NAME}" "${RUNNER_B}")
echo "  Teacher B job: ${job_b}"

echo "Submitting fused target build after teachers..."
job_build=$(sbatch --parsable --partition="${PARTITION}" \
  --dependency=afterok:${job_a}:${job_b} \
  --export=ALL,RUN_A_DIR="${RUN_A_DIR}",RUN_B_DIR="${RUN_B_DIR}",OUT_DIR="${TARGETS_DIR}" \
  "${RUNNER_BUILD}")
echo "  Build job: ${job_build}"

echo "Submitting offline fused-student training after build..."
job_student=$(sbatch --parsable --partition="${PARTITION}" \
  --dependency=afterok:${job_build} \
  --export=ALL,TARGETS_DIR="${TARGETS_DIR}",RUN_NAME="${STUDENT_RUN_NAME}" \
  "${RUNNER_STUDENT}")
echo "  Fused-student job: ${job_student}"

echo "Submitting Strategy-1 fused-KD after teachers..."
job_fusedkd=$(sbatch --parsable --partition="${PARTITION}" \
  --dependency=afterok:${job_a}:${job_b} \
  --export=ALL,RUN_NAME="${RUN_FUSEDKD_NAME}",TEACHER_A_RUN="${RUN_A_DIR}",TEACHER_B_RUN="${RUN_B_DIR}" \
  "${RUNNER_FUSEDKD}")
echo "  Fused-KD job: ${job_fusedkd}"

echo "Submitting Strategy-2 fusedstudent Stage-A after student..."
job_fusedstagea=$(sbatch --parsable --partition="${PARTITION}" \
  --dependency=afterok:${job_student} \
  --export=ALL,RUN_NAME="${RUN_FUSEDSTAGEA_NAME}",FUSED_STUDENT_CKPT="${STUDENT_CKPT}" \
  "${RUNNER_FUSEDSTAGEA}")
echo "  Fusedstudent Stage-A job: ${job_fusedstagea}"

echo "Submitting Strategy-3 fusedscore Stage-A after build+student..."
job_fusedscore=$(sbatch --parsable --partition="${PARTITION}" \
  --dependency=afterok:${job_build}:${job_student} \
  --export=ALL,RUN_NAME="${RUN_FUSEDSCORE_NAME}",RUN_A_DIR="${RUN_A_DIR}",RUN_B_DIR="${RUN_B_DIR}",FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ}",OFFLINE_TEACHER_CKPT="${STUDENT_CKPT}" \
  "${RUNNER_FUSEDSCORE}")
echo "  Fusedscore Stage-A job: ${job_fusedscore}"

echo "Submitting requested plain offline-teacher Stage-A run after student..."
job_offteach=$(sbatch --parsable --partition="${PARTITION}" \
  --dependency=afterok:${job_student} \
  --export=ALL,RUN_NAME="${RUN_OFFTEACH_NAME}",TEACHER_CKPT="${STUDENT_CKPT}" \
  "${RUNNER_OFFTEACH}")
echo "  Offline-teacher Stage-A job: ${job_offteach}"

echo "============================================================"
echo "Queued tier3 chain complete"
echo "Teacher A:                    ${job_a}"
echo "Teacher B:                    ${job_b}"
echo "Build fused targets:          ${job_build}"
echo "Offline fused-student:        ${job_student}"
echo "Strategy-1 fused-KD:          ${job_fusedkd}"
echo "Strategy-2 fusedstudentStageA:${job_fusedstagea}"
echo "Strategy-3 fusedscoreStageA:  ${job_fusedscore}"
echo "Extra offlineteacher Stage-A: ${job_offteach}"
echo "Run A dir:                    ${RUN_A_DIR}"
echo "Run B dir:                    ${RUN_B_DIR}"
echo "Fused targets:                ${FUSED_TARGETS_NPZ}"
echo "Student ckpt:                 ${STUDENT_CKPT}"
echo "============================================================"

