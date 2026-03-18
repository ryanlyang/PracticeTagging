#!/usr/bin/env bash
# Sweep runner for hlt_disagreement_risk_head.py
#
# Why this sweep:
# - target_tpr controls how strict the disagreement label is.
# - soft_alpha controls how aggressively risk suppresses top score.
# - risk_thresholds are already swept inside each run.
#
# Submit:
#   sbatch run_hlt_disagreement_risk_head_sweep_500k100.sh
#
# Optional overrides:
#   TARGET_TPR_VALUES_STR="0.3 0.5 0.7" SOFT_ALPHA_VALUES_STR="0.5 1.0 2.0" \
#   N_TRAIN_JETS=300000 MAX_CONSTITS=100 \
#   sbatch run_hlt_disagreement_risk_head_sweep_500k100.sh

#SBATCH -J rskHdSw
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 23:00:00
#SBATCH -o offline_reconstructor_logs/risk_head_sweeps/risk_head_sweep_%j.out
#SBATCH -e offline_reconstructor_logs/risk_head_sweeps/risk_head_sweep_%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-offline_reconstructor_logs/risk_head_sweeps}"
mkdir -p "${LOG_DIR}"

SAVE_DIR="${SAVE_DIR:-checkpoints/hlt_disagreement_risk_sweep}"
RUN_PREFIX="${RUN_PREFIX:-risk_head_500k100_sweep}"

TRAIN_PATH="${TRAIN_PATH:-./data}"
N_TRAIN_JETS="${N_TRAIN_JETS:-1000000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

TOP_EPOCHS="${TOP_EPOCHS:-60}"
TOP_PATIENCE="${TOP_PATIENCE:-15}"
TOP_LR="${TOP_LR:-5e-4}"

RISK_EPOCHS="${RISK_EPOCHS:-40}"
RISK_PATIENCE="${RISK_PATIENCE:-12}"
RISK_LR="${RISK_LR:-4e-4}"

# Swept knobs (recommended compact sweep).
TARGET_TPR_VALUES_STR="${TARGET_TPR_VALUES_STR:-0.30 0.50}"
SOFT_ALPHA_VALUES_STR="${SOFT_ALPHA_VALUES_STR:-1.0 2.0}"

# Threshold sweep is applied inside each run.
RISK_THRESHOLDS="${RISK_THRESHOLDS:-0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.92}"

IFS=' ' read -r -a TARGET_TPR_VALUES <<< "${TARGET_TPR_VALUES_STR}"
IFS=' ' read -r -a SOFT_ALPHA_VALUES <<< "${SOFT_ALPHA_VALUES_STR}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "$SLURM_SUBMIT_DIR"

sanitize() {
  echo "$1" | sed 's/-/m/g; s/\./p/g; s/+//g'
}

summary_file="${LOG_DIR}/risk_head_sweep_summary_${SLURM_JOB_ID}.tsv"
cat > "${summary_file}" <<'TSV'
cfg_idx	run_name	status	target_tpr	soft_alpha	baseline_auc	baseline_fpr30	baseline_fpr50	best_hard_tau_fpr30	best_hard_fpr30	best_hard_tau_fpr50	best_hard_fpr50	best_soft_tau_fpr30	best_soft_fpr30	best_soft_tau_fpr50	best_soft_fpr50	risk_target_rate_train	risk_target_rate_val	risk_target_rate_test	risk_head_auc	risk_head_ap
TSV

cfg_idx=0
total_cfgs=$(( ${#TARGET_TPR_VALUES[@]} * ${#SOFT_ALPHA_VALUES[@]} ))

echo "============================================================"
echo "Risk-head sweep: ${total_cfgs} configs"
echo "Save dir: ${SAVE_DIR}"
echo "Run prefix: ${RUN_PREFIX}"
echo "N jets / max constits: ${N_TRAIN_JETS} / ${MAX_CONSTITS}"
echo "Risk thresholds: ${RISK_THRESHOLDS}"
echo "Summary: ${summary_file}"
echo "============================================================"

for target_tpr in "${TARGET_TPR_VALUES[@]}"; do
  for soft_alpha in "${SOFT_ALPHA_VALUES[@]}"; do
    run_name="${RUN_PREFIX}_cfg$(printf '%02d' "${cfg_idx}")_tpr$(sanitize "${target_tpr}")_a$(sanitize "${soft_alpha}")"
    metrics_path="${SAVE_DIR}/${run_name}/risk_head_metrics.json"

    echo
    echo ">>> cfg ${cfg_idx}/$((total_cfgs-1))"
    echo "    run_name=${run_name}"
    echo "    target_tpr=${target_tpr}, soft_alpha=${soft_alpha}"

    cmd=(
      python hlt_disagreement_risk_head.py
      --train_path "${TRAIN_PATH}"
      --save_dir "${SAVE_DIR}"
      --run_name "${run_name}"
      --n_train_jets "${N_TRAIN_JETS}"
      --offset_jets "${OFFSET_JETS}"
      --max_constits "${MAX_CONSTITS}"
      --num_workers "${NUM_WORKERS}"
      --seed "${SEED}"
      --device "${DEVICE}"
      --top_epochs "${TOP_EPOCHS}"
      --top_patience "${TOP_PATIENCE}"
      --top_lr "${TOP_LR}"
      --risk_epochs "${RISK_EPOCHS}"
      --risk_patience "${RISK_PATIENCE}"
      --risk_lr "${RISK_LR}"
      --target_tpr "${target_tpr}"
      --risk_thresholds "${RISK_THRESHOLDS}"
      --soft_alpha "${soft_alpha}"
    )

    set +e
    "${cmd[@]}"
    rc=$?
    set -e

    if [[ "${rc}" -eq 0 && -f "${metrics_path}" ]]; then
      metrics_row="$(
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
    get(m, "top_tag_test", "baseline", "auc"),
    get(m, "top_tag_test", "baseline", "fpr30"),
    get(m, "top_tag_test", "baseline", "fpr50"),
    get(m, "top_tag_test", "best_hard_by_fpr30", "tau"),
    get(m, "top_tag_test", "best_hard_by_fpr30", "hard", "fpr30"),
    get(m, "top_tag_test", "best_hard_by_fpr50", "tau"),
    get(m, "top_tag_test", "best_hard_by_fpr50", "hard", "fpr50"),
    get(m, "top_tag_test", "best_soft_by_fpr30", "tau"),
    get(m, "top_tag_test", "best_soft_by_fpr30", "soft", "fpr30"),
    get(m, "top_tag_test", "best_soft_by_fpr50", "tau"),
    get(m, "top_tag_test", "best_soft_by_fpr50", "soft", "fpr50"),
    get(m, "risk_target_rate", "train"),
    get(m, "risk_target_rate", "val"),
    get(m, "risk_target_rate", "test"),
    get(m, "risk_head_test_target_metrics", "auc"),
    get(m, "risk_head_test_target_metrics", "ap"),
]
print("\t".join(out))
PY
      )"
    else
      metrics_row=$'fail\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan'
    fi

    IFS=$'\t' read -r status \
      baseline_auc baseline_fpr30 baseline_fpr50 \
      best_hard_tau_fpr30 best_hard_fpr30 \
      best_hard_tau_fpr50 best_hard_fpr50 \
      best_soft_tau_fpr30 best_soft_fpr30 \
      best_soft_tau_fpr50 best_soft_fpr50 \
      risk_target_rate_train risk_target_rate_val risk_target_rate_test \
      risk_head_auc risk_head_ap \
      <<< "${metrics_row}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${cfg_idx}" "${run_name}" "${status}" "${target_tpr}" "${soft_alpha}" \
      "${baseline_auc}" "${baseline_fpr30}" "${baseline_fpr50}" \
      "${best_hard_tau_fpr30}" "${best_hard_fpr30}" \
      "${best_hard_tau_fpr50}" "${best_hard_fpr50}" \
      "${best_soft_tau_fpr30}" "${best_soft_fpr30}" \
      "${best_soft_tau_fpr50}" "${best_soft_fpr50}" \
      "${risk_target_rate_train}" "${risk_target_rate_val}" "${risk_target_rate_test}" \
      "${risk_head_auc}" "${risk_head_ap}" \
      >> "${summary_file}"

    cfg_idx=$((cfg_idx + 1))
  done
done

echo
echo "============================================================"
echo "Sweep complete. Summary:"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${summary_file}"
else
  cat "${summary_file}"
fi
echo "Summary written to: ${summary_file}"
