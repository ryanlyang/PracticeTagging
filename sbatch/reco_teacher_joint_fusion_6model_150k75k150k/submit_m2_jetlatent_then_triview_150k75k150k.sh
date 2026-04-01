#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN1="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m2_joint_delta005_fulltrain_prog_unfreeze_jetlatent_set2set_150k75k150k.sh"
RUN2="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m2_triview_jetlatent_m2ref_frozen_then_joint_150k75k150k.sh"

if [[ ! -f "$RUN1" ]]; then
  echo "Missing runner: $RUN1" >&2
  exit 1
fi
if [[ ! -f "$RUN2" ]]; then
  echo "Missing runner: $RUN2" >&2
  exit 1
fi

echo "Submitting jet-latent set2set run..."
SUB1_OUT=$(sbatch "$RUN1")
echo "$SUB1_OUT"
JID1=$(echo "$SUB1_OUT" | awk '{print $4}')
if [[ -z "${JID1}" ]]; then
  echo "Failed to parse first job id" >&2
  exit 1
fi

echo "Submitting tri-view run with dependency afterok:${JID1}..."
SUB2_OUT=$(sbatch --dependency=afterok:${JID1} "$RUN2")
echo "$SUB2_OUT"
JID2=$(echo "$SUB2_OUT" | awk '{print $4}')
if [[ -z "${JID2}" ]]; then
  echo "Failed to parse second job id" >&2
  exit 1
fi

echo ""
echo "Dependency chain created:"
echo "  JetLatent job id : ${JID1}"
echo "  TriView job id   : ${JID2} (afterok:${JID1})"
