#!/usr/bin/env bash
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

S_M2="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m2_joint_delta005_150k75k150k.sh"
S_M3="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m3_recoteacher_s09_150k75k150k.sh"
S_M4="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_150k75k150k.sh"
S_M5="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m5_joint_s01_full_150k75k150k.sh"
S_M6="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m6_concat_stagea_corrected_150k75k150k.sh"

S_M7="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m7_stageA_residual_hlt_150k75k150k.sh"
S_M8="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m8_direct_residual_ablation_150k75k150k.sh"
S_M9L="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m9_stageA_residual_hlt_offdrop_low_150k75k150k.sh"
S_M9M="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m9_stageA_residual_hlt_offdrop_mid_150k75k150k.sh"
S_M9H="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m9_stageA_residual_hlt_offdrop_high_150k75k150k.sh"

S_M4K40="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_40max_150k75k150k.sh"
S_M4K60="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_60max_150k75k150k.sh"
S_M4K80="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_80max_150k75k150k.sh"

S_M10="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m10_recoteacher_s01_corrected_antioverlap_150k75k150k.sh"
S_M11="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m11_recoteacher_s01_corrected_feat_noangle_150k75k150k.sh"
S_M12="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m12_recoteacher_s01_corrected_feat_noscale_150k75k150k.sh"
S_M13="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m13_recoteacher_s01_corrected_feat_coreshape_150k75k150k.sh"

S_AN="sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_analyze_hlt_joint18_fusion_150k75k150k.sh"

BASE_DIR="${BASE_DIR:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k}"
M2_RUN_DIR="${M2_RUN_DIR:-${BASE_DIR}/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0}"
M3_RUN_DIR="${M3_RUN_DIR:-${BASE_DIR}/model3_recoteacher_s09/model3_recoteacher_s09_150k75k150k_seed0}"

# Default behavior: use existing m2/m3 checkpoints, do not resubmit m2/m3 jobs.
USE_EXISTING_M2_M3="${USE_EXISTING_M2_M3:-1}"

for s in \
  "$S_M2" "$S_M3" "$S_M4" "$S_M5" "$S_M6" \
  "$S_M7" "$S_M8" "$S_M9L" "$S_M9M" "$S_M9H" \
  "$S_M4K40" "$S_M4K60" "$S_M4K80" \
  "$S_M10" "$S_M11" "$S_M12" "$S_M13" \
  "$S_AN"
  do
  if [[ ! -f "$s" ]]; then
    echo "Missing script: $s" >&2
    exit 1
  fi
done

if [[ "${USE_EXISTING_M2_M3}" == "1" ]]; then
  j_m2="existing"
  j_m3="existing"
  echo "Using existing m2/m3 run dirs for analysis:"
  echo "  M2_RUN_DIR=${M2_RUN_DIR}"
  echo "  M3_RUN_DIR=${M3_RUN_DIR}"
else
  j_m2=$(sbatch "$S_M2" | awk '{print $4}')
  j_m3=$(sbatch "$S_M3" | awk '{print $4}')
fi

j_m4=$(sbatch "$S_M4" | awk '{print $4}')
j_m5=$(sbatch "$S_M5" | awk '{print $4}')
j_m6=$(sbatch "$S_M6" | awk '{print $4}')

j_m7=$(sbatch "$S_M7" | awk '{print $4}')
j_m8=$(sbatch "$S_M8" | awk '{print $4}')
j_m9l=$(sbatch "$S_M9L" | awk '{print $4}')
j_m9m=$(sbatch "$S_M9M" | awk '{print $4}')
j_m9h=$(sbatch "$S_M9H" | awk '{print $4}')

j_m4k40=$(sbatch "$S_M4K40" | awk '{print $4}')
j_m4k60=$(sbatch "$S_M4K60" | awk '{print $4}')
j_m4k80=$(sbatch "$S_M4K80" | awk '{print $4}')

j_m10=$(sbatch "$S_M10" | awk '{print $4}')
j_m11=$(sbatch "$S_M11" | awk '{print $4}')
j_m12=$(sbatch "$S_M12" | awk '{print $4}')
j_m13=$(sbatch "$S_M13" | awk '{print $4}')

deps_base="${j_m4}:${j_m5}:${j_m6}:${j_m7}:${j_m8}:${j_m9l}:${j_m9m}:${j_m9h}:${j_m4k40}:${j_m4k60}:${j_m4k80}:${j_m10}:${j_m11}:${j_m12}:${j_m13}"
if [[ "${USE_EXISTING_M2_M3}" == "1" ]]; then
  deps="${deps_base}"
else
  deps="${j_m2}:${j_m3}:${deps_base}"
fi

j_an=$(
  sbatch \
    --dependency=afterok:${deps} \
    --export=ALL,M2_RUN_DIR="${M2_RUN_DIR}",M3_RUN_DIR="${M3_RUN_DIR}" \
    "$S_AN" | awk '{print $4}'
)

echo "Submitted jobs:"
echo "  m2  = ${j_m2}"
echo "  m3  = ${j_m3}"
echo "  m4  = ${j_m4}"
echo "  m5  = ${j_m5}"
echo "  m6  = ${j_m6}"
echo "  m7  = ${j_m7}"
echo "  m8  = ${j_m8}"
echo "  m9L = ${j_m9l}"
echo "  m9M = ${j_m9m}"
echo "  m9H = ${j_m9h}"
echo "  k40 = ${j_m4k40}"
echo "  k60 = ${j_m4k60}"
echo "  k80 = ${j_m4k80}"
echo "  m10 = ${j_m10}"
echo "  m11 = ${j_m11}"
echo "  m12 = ${j_m12}"
echo "  m13 = ${j_m13}"
echo "Submitted 18-model analysis: ${j_an}"
echo "  using M2_RUN_DIR=${M2_RUN_DIR}"
echo "  using M3_RUN_DIR=${M3_RUN_DIR}"
echo "  dependency=afterok:${deps}"
