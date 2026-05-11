#!/usr/bin/env bash
set -euo pipefail

# Queue three focused 250k fused-KD recovery ablations.
# Optional:
#   BASE_DEPENDENCY=afterok:<jobid> PARTITION=tier3 bash submit_jetclass_250k_fusedkd_recovery_ablation3.sh

PARTITION="${PARTITION:-tier3}"
BASE_DEPENDENCY="${BASE_DEPENDENCY:-}"

RUNNER="${RUNNER:-run_train_jetclass_joint_dualview_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_two_teacher.sh}"
SAVE_DIR="${SAVE_DIR:-checkpoints/jetclass_joint_dualview}"
TEACHER_A_RUN="${TEACHER_A_RUN:-${SAVE_DIR}/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_path_gentok56_ablate_lcons003_recoonlydual}"
TEACHER_B_RUN="${TEACHER_B_RUN:-${SAVE_DIR}/jetclass_joint_confgen_v2attr_250k50k250k_stronger_canonical_v1hlt_hltplus25_gentok56}"

[[ -f "${RUNNER}" ]] || { echo "Missing runner: ${RUNNER}" >&2; exit 1; }

dep_args=()
if [[ -n "${BASE_DEPENDENCY}" ]]; then
  dep_args=(--dependency="${BASE_DEPENDENCY}")
fi

submit_one() {
  local run_name="$1"
  shift
  echo "Submitting ${run_name} ..."
  local jid
  jid=$(sbatch --parsable --partition="${PARTITION}" "${dep_args[@]}" \
    --export=ALL,RUN_NAME="${run_name}",TEACHER_A_RUN="${TEACHER_A_RUN}",TEACHER_B_RUN="${TEACHER_B_RUN}",$* \
    "${RUNNER}")
  echo "  -> ${jid}"
}

# A1) Remove reco-only auxiliary stage to avoid over-specializing on noisy reconstructed view.
submit_one \
  "jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_a1_norecoonly" \
  "ENABLE_RECO_ONLY_AFTER_STAGEA=0,DISTILL_WEIGHT_A=0.50,DISTILL_TEMP=2.5,DISTILL_ALPHA_KL=1.0,DISTILL_ALPHA_CE=0.25,DISTILL_PHASE2_LAMBDA_RECO=0.20,DISTILL_PHASE2_LAMBDA_CONS=0.03"

# A2) Stronger reconstruction pressure during distill phase-2.
submit_one \
  "jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_a2_recostrong" \
  "ENABLE_RECO_ONLY_AFTER_STAGEA=1,DISTILL_WEIGHT_A=0.50,DISTILL_TEMP=2.5,DISTILL_ALPHA_KL=1.0,DISTILL_ALPHA_CE=0.25,DISTILL_PHASE2_LAMBDA_RECO=0.35,DISTILL_PHASE2_LAMBDA_CONS=0.08"

# A3) More teacher-driven KD (less hard-label CE), with reco-only disabled.
submit_one \
  "jetclass_joint_confgen_v2attr_250k50k250k_v1hltplus25_fusedkd_a3_kdstrong_norecoonly" \
  "ENABLE_RECO_ONLY_AFTER_STAGEA=0,DISTILL_WEIGHT_A=0.50,DISTILL_TEMP=2.0,DISTILL_ALPHA_KL=1.0,DISTILL_ALPHA_CE=0.10,DISTILL_PHASE2_LAMBDA_RECO=0.20,DISTILL_PHASE2_LAMBDA_CONS=0.03"

echo "Done."

