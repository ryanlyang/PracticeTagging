#!/usr/bin/env bash
#SBATCH --job-name=ajrfus
#SBATCH --partition=debug
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_debug/fusion_analysis_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_debug/fusion_analysis_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_debug

STAGEA_SAVE_DIR="${STAGEA_SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_debug/stagea}"
STAGEA_RUN_NAME="${STAGEA_RUN_NAME:-recoteacher_stageAonly_s09delta_75k25k150k_seed0}"
JOINT_SAVE_DIR="${JOINT_SAVE_DIR:-checkpoints/reco_teacher_joint_fusion_debug/joint}"
JOINT_RUN_NAME="${JOINT_RUN_NAME:-joint_unmergeonly_75k25k150k_seed0}"
TARGET_TPR="${TARGET_TPR:-0.50}"
SEED="${SEED:-0}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

STAGEA_RUN_DIR="${STAGEA_SAVE_DIR}/${STAGEA_RUN_NAME}"
JOINT_RUN_DIR="${JOINT_SAVE_DIR}/${JOINT_RUN_NAME}"

CMD=(
  python analyze_hlt_joint_recoteacher_fusion.py
  --stagea_run_dir "${STAGEA_RUN_DIR}"
  --joint_run_dir "${JOINT_RUN_DIR}"
  --target_tpr "${TARGET_TPR}"
  --weight_step_2 0.01
  --weight_step_3 0.02
  --meta_sel_frac 0.30
  --meta_c_grid "0.05,0.1,0.3,1,3,10,30"
  --seed "${SEED}"
  --output_name "fusion_hlt_joint_recoteacher_analysis.json"
)

echo "============================================================"
echo "Fusion analysis: HLT + Joint + RecoTeacher"
echo "StageA run: ${STAGEA_RUN_DIR}"
echo "Joint run : ${JOINT_RUN_DIR}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Done analysis: ${JOINT_RUN_DIR}/fusion_hlt_joint_recoteacher_analysis.json"
