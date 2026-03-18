#!/usr/bin/env bash
# Sweep 10 discrepancy strengths with RECONSTRUCTION-ONLY weighting (Stage C).
#
# Submit:
#   sbatch run_finetune_stagec_from_stage2_discrepancy_recoonly_strength_sweep_300k100.sh

#SBATCH -J stgCDRw
#SBATCH -p tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o offline_reconstructor_logs/stagec_discrepancy_sweeps_recoonly/stagec_disc_recoonly_%j.out
#SBATCH -e offline_reconstructor_logs/stagec_discrepancy_sweeps_recoonly/stagec_disc_recoonly_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/stagec_discrepancy_sweeps_recoonly}"
mkdir -p "${LOG_DIR}"

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_300kJ100C}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_discrepancy}"
RUN_PREFIX="${RUN_PREFIX:-stagec_discw_recoonly_300k100_strength}"

N_TRAIN_JETS="${N_TRAIN_JETS:-300000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-65}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
STAGEC_LR_DUAL="${STAGEC_LR_DUAL:-1e-5}"
STAGEC_LR_RECO="${STAGEC_LR_RECO:-5e-6}"
LAMBDA_RECO="${LAMBDA_RECO:-0.4}"
LAMBDA_CONS="${LAMBDA_CONS:-0.06}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"

DISC_WEIGHT_MODE="${DISC_WEIGHT_MODE:-smooth_delta}"
DISC_TARGET_TPR="${DISC_TARGET_TPR:-0.50}"
DISC_TEACHER_CONF_MIN="${DISC_TEACHER_CONF_MIN:-0.65}"
DISC_CORRECTNESS_TAU="${DISC_CORRECTNESS_TAU:-0.05}"
DISC_DISABLE_TEACHER_HARD_CORRECT_GATE="${DISC_DISABLE_TEACHER_HARD_CORRECT_GATE:-0}"
DISC_DISABLE_TEACHER_CONF_GATE="${DISC_DISABLE_TEACHER_CONF_GATE:-1}"
DISC_DISABLE_TEACHER_BETTER_GATE="${DISC_DISABLE_TEACHER_BETTER_GATE:-1}"
DISC_INCLUDE_POS="${DISC_INCLUDE_POS:-0}"
DISC_POS_SCALE="${DISC_POS_SCALE:-0.25}"
DISC_APPLY_TO_RECO="${DISC_APPLY_TO_RECO:-1}"
DISC_DISABLE_CLS_WEIGHT="${DISC_DISABLE_CLS_WEIGHT:-1}"

# Heavy reco-only presets in requested order (no control run in this sweep):
# 50,100 | 15,20(nonorm) | 100,200 | 12,15 | 10,12 | 8,10 | 6,8 | 4,6 | 3,5 | 2,4 | 1,3
DISC_ENABLE=(1 1 1 1 1 1 1 1 1 1 1)
DISC_LAMBDA=(50.0 15.0 100.0 12.0 10.0 8.0 6.0 4.0 3.0 2.0 1.0)
DISC_MAX_MULT=(100.0 20.0 200.0 15.0 12.0 10.0 8.0 6.0 5.0 4.0 3.0)
DISC_TAU=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)
DISC_NO_MEAN_NORM=(0 1 0 0 0 0 0 0 0 0 0)

N_CFG="${#DISC_ENABLE[@]}"
if [[ "${N_CFG}" -ne "${#DISC_LAMBDA[@]}" || "${N_CFG}" -ne "${#DISC_MAX_MULT[@]}" || "${N_CFG}" -ne "${#DISC_TAU[@]}" || "${N_CFG}" -ne "${#DISC_NO_MEAN_NORM[@]}" ]]; then
  echo "Config array length mismatch." >&2
  exit 2
fi

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

sanitize() {
  echo "$1" | sed 's/-/m/g; s/\./p/g; s/+//g'
}

SUMMARY_FILE="${LOG_DIR}/stagec_disc_recoonly_summary_${SLURM_JOB_ID}.tsv"
cat > "${SUMMARY_FILE}" <<'TSV'
cfg_idx	run_name	status	disc_enable	disc_lambda	disc_max_mult	disc_tau	disc_no_mean_norm	mean_weight	p95_weight	frac_w_gt_1p5	stage2_auc	stage2_fpr30	stage2_fpr50	stagec_auc	stagec_fpr30	stagec_fpr50	stagec_bestfpr50_auc	stagec_bestfpr50_fpr30	stagec_bestfpr50_fpr50	baseline_auc	baseline_fpr30	baseline_fpr50	teacher_auc	teacher_fpr30	teacher_fpr50
TSV

echo "============================================================"
echo "Stage-C discrepancy RECO-only strength sweep"
echo "Run dir: ${RUN_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Configs: ${N_CFG}"
echo "Discrepancy mode: ${DISC_WEIGHT_MODE}"
echo "Teacher gates: hard=$((1-DISC_DISABLE_TEACHER_HARD_CORRECT_GATE)) conf=$((1-DISC_DISABLE_TEACHER_CONF_GATE)) better=$((1-DISC_DISABLE_TEACHER_BETTER_GATE))"
echo "Teacher conf min/tau: ${DISC_TEACHER_CONF_MIN} / ${DISC_CORRECTNESS_TAU}"
echo "Include pos branch: ${DISC_INCLUDE_POS} (pos_scale=${DISC_POS_SCALE})"
echo "Apply weights: cls=$((1-DISC_DISABLE_CLS_WEIGHT)) reco=${DISC_APPLY_TO_RECO}"
echo "Summary: ${SUMMARY_FILE}"
echo "============================================================"

for ((i=0; i<N_CFG; i++)); do
  de="${DISC_ENABLE[$i]}"
  dl="${DISC_LAMBDA[$i]}"
  dm="${DISC_MAX_MULT[$i]}"
  dt="${DISC_TAU[$i]}"
  dnn="${DISC_NO_MEAN_NORM[$i]}"

  run_name="${RUN_PREFIX}_cfg$(printf '%02d' "${i}")_de${de}_lam$(sanitize "${dl}")_max$(sanitize "${dm}")_tau$(sanitize "${dt}")"
  if [[ "${dnn}" -eq 1 ]]; then
    run_name="${run_name}_nonorm"
  fi

  metrics_path="${SAVE_DIR}/${run_name}/stagec_refine_metrics.json"

  echo
  echo ">>> cfg ${i}/$((N_CFG-1)) | ${run_name}"

  cmd=(
    python finetune_stagec_from_stage2_discrepancy_weighted.py
    --run_dir "${RUN_DIR}"
    --save_dir "${SAVE_DIR}"
    --run_name "${run_name}"
    --n_train_jets "${N_TRAIN_JETS}"
    --offset_jets "${OFFSET_JETS}"
    --max_constits "${MAX_CONSTITS}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --stageC_epochs "${STAGEC_EPOCHS}"
    --stageC_patience "${STAGEC_PATIENCE}"
    --stageC_min_epochs "${STAGEC_MIN_EPOCHS}"
    --stageC_lr_dual "${STAGEC_LR_DUAL}"
    --stageC_lr_reco "${STAGEC_LR_RECO}"
    --lambda_reco "${LAMBDA_RECO}"
    --lambda_cons "${LAMBDA_CONS}"
    --selection_metric "${SELECTION_METRIC}"
    --device "${DEVICE}"
    --disc_target_tpr "${DISC_TARGET_TPR}"
    --disc_weight_mode "${DISC_WEIGHT_MODE}"
    --disc_lambda "${dl}"
    --disc_max_mult "${dm}"
    --disc_tau "${dt}"
    --disc_teacher_conf_min "${DISC_TEACHER_CONF_MIN}"
    --disc_correctness_tau "${DISC_CORRECTNESS_TAU}"
  )

  if [[ "${de}" -eq 1 ]]; then
    cmd+=(--disc_weight_enable)
  fi
  if [[ "${dnn}" -eq 1 ]]; then
    cmd+=(--disc_no_mean_normalize)
  fi
  if [[ "${DISC_DISABLE_TEACHER_HARD_CORRECT_GATE}" -eq 1 ]]; then
    cmd+=(--disc_disable_teacher_hard_correct_gate)
  fi
  if [[ "${DISC_DISABLE_TEACHER_CONF_GATE}" -eq 1 ]]; then
    cmd+=(--disc_disable_teacher_conf_gate)
  fi
  if [[ "${DISC_DISABLE_TEACHER_BETTER_GATE}" -eq 1 ]]; then
    cmd+=(--disc_disable_teacher_better_gate)
  fi
  if [[ "${DISC_INCLUDE_POS}" -eq 1 ]]; then
    cmd+=(--disc_include_pos --disc_pos_scale "${DISC_POS_SCALE}")
  fi
  if [[ "${DISC_APPLY_TO_RECO}" -eq 1 ]]; then
    cmd+=(--disc_apply_to_reco)
  fi
  if [[ "${DISC_DISABLE_CLS_WEIGHT}" -eq 1 ]]; then
    cmd+=(--disc_disable_cls_weight)
  fi

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 && -f "${metrics_path}" ]]; then
    row="$(
      python3 - "${metrics_path}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)

def get(d, *ks):
    x = d
    for k in ks:
        if not isinstance(x, dict) or k not in x:
            return "nan"
        x = x[k]
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "nan"

out = [
    "ok",
    get(m, "stageC_discrepancy_weighting", "mean_weight"),
    get(m, "stageC_discrepancy_weighting", "p95_weight"),
    get(m, "stageC_discrepancy_weighting", "fraction_w_gt_1p5"),
    get(m, "test_stage2_loaded", "auc"),
    get(m, "test_stage2_loaded", "fpr30"),
    get(m, "test_stage2_loaded", "fpr50"),
    get(m, "test_stageC_selected", "auc"),
    get(m, "test_stageC_selected", "fpr30"),
    get(m, "test_stageC_selected", "fpr50"),
    get(m, "test_stageC_bestfpr50", "auc"),
    get(m, "test_stageC_bestfpr50", "fpr30"),
    get(m, "test_stageC_bestfpr50", "fpr50"),
    get(m, "test_baseline_loaded", "auc"),
    get(m, "test_baseline_loaded", "fpr30"),
    get(m, "test_baseline_loaded", "fpr50"),
    get(m, "test_teacher_loaded", "auc"),
    get(m, "test_teacher_loaded", "fpr30"),
    get(m, "test_teacher_loaded", "fpr50"),
]
print("\t".join(out))
PY
    )"
  else
    row=$'fail\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan'
  fi

  IFS=$'\t' read -r status \
    mean_weight p95_weight frac_w_gt_1p5 \
    stage2_auc stage2_fpr30 stage2_fpr50 \
    stagec_auc stagec_fpr30 stagec_fpr50 \
    stagec_bestfpr50_auc stagec_bestfpr50_fpr30 stagec_bestfpr50_fpr50 \
    baseline_auc baseline_fpr30 baseline_fpr50 \
    teacher_auc teacher_fpr30 teacher_fpr50 \
    <<< "${row}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${i}" "${run_name}" "${status}" \
    "${de}" "${dl}" "${dm}" "${dt}" "${dnn}" \
    "${mean_weight}" "${p95_weight}" "${frac_w_gt_1p5}" \
    "${stage2_auc}" "${stage2_fpr30}" "${stage2_fpr50}" \
    "${stagec_auc}" "${stagec_fpr30}" "${stagec_fpr50}" \
    "${stagec_bestfpr50_auc}" "${stagec_bestfpr50_fpr30}" "${stagec_bestfpr50_fpr50}" \
    "${baseline_auc}" "${baseline_fpr30}" "${baseline_fpr50}" \
    "${teacher_auc}" "${teacher_fpr30}" "${teacher_fpr50}" \
    >> "${SUMMARY_FILE}"
done

echo
echo "============================================================"
echo "Sweep complete. Summary:"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${SUMMARY_FILE}"
else
  cat "${SUMMARY_FILE}"
fi
echo "Summary written to: ${SUMMARY_FILE}"
