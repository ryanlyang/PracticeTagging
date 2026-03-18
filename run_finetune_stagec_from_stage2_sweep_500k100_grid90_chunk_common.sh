#!/usr/bin/env bash
# 90-config Stage-C sweep from saved Stage2 (500k/100), split across 9 chunks.
#
# Grid:
#   lambda_reco         in {0.1, 0.4, 0.7}
#   stageC_lr_dual      in {1e-6, 1e-5, 5e-5}
#   stageC_lr_reco      in {5e-7, 5e-6, 5e-5, 5e-4, 5e-3}
#   stageC_lambda_rank  in {0.0, 0.1}
#   lambda_cons fixed at 0.06 (override via env var LAMBDA_CONS)
#
# Total configs: 2 * 3 * 3 * 5 = 90
# Chunking: CHUNK_ID in 0..8, each chunk runs 10 configs.

set -euo pipefail

mkdir -p offline_reconstructor_logs

CHUNK_ID="${CHUNK_ID:-}"
if [[ -z "${CHUNK_ID}" ]]; then
  echo "Error: CHUNK_ID must be set (0..8)."
  exit 2
fi
if ! [[ "${CHUNK_ID}" =~ ^[0-8]$ ]]; then
  echo "Error: invalid CHUNK_ID='${CHUNK_ID}' (expected 0..8)."
  exit 2
fi

RUN_DIR="${RUN_DIR:-checkpoints/offline_reconstructor_joint/joint_100k_80c_stage2save_auc_norankc_nopriv_unmergeonly_rho090_500kJ100C}"
SAVE_DIR="${SAVE_DIR:-checkpoints/offline_reconstructor_joint_stagec_refine_sweep_500k100_grid90}"
SWEEP_TAG="${SWEEP_TAG:-stagec_sweep_500k100_grid90}"

# Keep these aligned with the source run unless intentionally overridden.
N_TRAIN_JETS="${N_TRAIN_JETS:-500000}"
OFFSET_JETS="${OFFSET_JETS:-0}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

STAGEC_EPOCHS="${STAGEC_EPOCHS:-65}"
STAGEC_PATIENCE="${STAGEC_PATIENCE:-14}"
STAGEC_MIN_EPOCHS="${STAGEC_MIN_EPOCHS:-25}"
SELECTION_METRIC="${SELECTION_METRIC:-auc}"
CORRECTED_WEIGHT_FLOOR="${CORRECTED_WEIGHT_FLOOR:-1e-4}"
USE_CORRECTED_FLAGS="${USE_CORRECTED_FLAGS:-0}"

LAMBDA_CONS="${LAMBDA_CONS:-0.06}"

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

mapfile -t CONFIGS < <(
  python3 - <<'PY'
lambda_reco_vals = [0.1, 0.4, 0.7]
lr_dual_vals = [1e-6, 1e-5, 5e-5]
lr_reco_vals = [5e-7, 5e-6, 5e-5, 5e-4, 5e-3]
lambda_rank_vals = [0.0, 0.1]

idx = 0
for lambda_rank in lambda_rank_vals:
    for lambda_reco in lambda_reco_vals:
        for lr_dual in lr_dual_vals:
            for lr_reco in lr_reco_vals:
                print(f"{idx}|{lambda_rank}|{lambda_reco}|{lr_dual}|{lr_reco}")
                idx += 1
PY
)

if [[ "${#CONFIGS[@]}" -ne 90 ]]; then
  echo "Error: expected 90 configs, got ${#CONFIGS[@]}"
  exit 2
fi

start=$(( CHUNK_ID * 10 ))
end=$(( start + 10 ))

summary_file="offline_reconstructor_logs/stagec_sweep_500k100_grid90_chunk$(printf '%02d' $((CHUNK_ID + 1)))_summary.tsv"
cat > "${summary_file}" <<'TSV'
cfg_idx	run_name	status	stageC_lambda_rank	lambda_reco	lambda_cons	stageC_lr_dual	stageC_lr_reco	stage2_auc	stage2_fpr30	stage2_fpr50	stagec_auc	stagec_fpr30	stagec_fpr50	stagec_bestfpr50_auc	stagec_bestfpr50_fpr30	stagec_bestfpr50_fpr50	baseline_auc	baseline_fpr30	baseline_fpr50	teacher_auc	teacher_fpr30	teacher_fpr50
TSV

echo "============================================================"
echo "Stage-C sweep chunk ${CHUNK_ID} | range [${start}, ${end})"
echo "Run dir: ${RUN_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Sweep tag: ${SWEEP_TAG}"
echo "Fixed lambda_cons: ${LAMBDA_CONS}"
echo "Summary file: ${summary_file}"
echo "============================================================"

for ((i=start; i<end; i++)); do
  IFS='|' read -r cfg_idx stagec_lambda_rank lambda_reco stagec_lr_dual stagec_lr_reco <<< "${CONFIGS[i]}"

  run_name="${SWEEP_TAG}_cfg$(printf '%03d' "${cfg_idx}")_lrank$(sanitize "${stagec_lambda_rank}")_lreco$(sanitize "${lambda_reco}")_lcons$(sanitize "${LAMBDA_CONS}")_lrd$(sanitize "${stagec_lr_dual}")_lrr$(sanitize "${stagec_lr_reco}")"
  metrics_path="${SAVE_DIR}/${run_name}/stagec_refine_metrics.json"

  echo
  echo ">>> [chunk ${CHUNK_ID}] cfg=${cfg_idx}"
  echo "    run_name=${run_name}"
  echo "    lrank=${stagec_lambda_rank} lreco=${lambda_reco} lcons=${LAMBDA_CONS} lrd=${stagec_lr_dual} lrr=${stagec_lr_reco}"

  cmd=(
    python finetune_stagec_from_stage2.py
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
    --stageC_lr_dual "${stagec_lr_dual}"
    --stageC_lr_reco "${stagec_lr_reco}"
    --stageC_lambda_rank "${stagec_lambda_rank}"
    --lambda_reco "${lambda_reco}"
    --lambda_cons "${LAMBDA_CONS}"
    --selection_metric "${SELECTION_METRIC}"
    --corrected_weight_floor "${CORRECTED_WEIGHT_FLOOR}"
    --device cuda
  )
  if [[ "${USE_CORRECTED_FLAGS}" == "1" ]]; then
    cmd+=(--use_corrected_flags)
  fi

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 && -f "${metrics_path}" ]]; then
    metrics_row="$(
      python3 - "${metrics_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)

def f(d, k):
    if not isinstance(d, dict):
        return "nan"
    v = d.get(k, None)
    if v is None:
        return "nan"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "nan"

s2 = m.get("test_stage2_loaded", {})
sj = m.get("test_stageC_selected", {})
sf = m.get("test_stageC_bestfpr50", {})
tb = m.get("test_baseline_loaded", {})
tt = m.get("test_teacher_loaded", {})

print("\t".join([
    "ok",
    f(s2, "auc"), f(s2, "fpr30"), f(s2, "fpr50"),
    f(sj, "auc"), f(sj, "fpr30"), f(sj, "fpr50"),
    f(sf, "auc"), f(sf, "fpr30"), f(sf, "fpr50"),
    f(tb, "auc"), f(tb, "fpr30"), f(tb, "fpr50"),
    f(tt, "auc"), f(tt, "fpr30"), f(tt, "fpr50"),
]))
PY
    )"
  else
    metrics_row=$'fail\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan'
  fi

  IFS=$'\t' read -r status \
    stage2_auc stage2_fpr30 stage2_fpr50 \
    stagec_auc stagec_fpr30 stagec_fpr50 \
    stagec_bestfpr50_auc stagec_bestfpr50_fpr30 stagec_bestfpr50_fpr50 \
    baseline_auc baseline_fpr30 baseline_fpr50 \
    teacher_auc teacher_fpr30 teacher_fpr50 \
    <<< "${metrics_row}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${cfg_idx}" "${run_name}" "${status}" \
    "${stagec_lambda_rank}" "${lambda_reco}" "${LAMBDA_CONS}" "${stagec_lr_dual}" "${stagec_lr_reco}" \
    "${stage2_auc}" "${stage2_fpr30}" "${stage2_fpr50}" \
    "${stagec_auc}" "${stagec_fpr30}" "${stagec_fpr50}" \
    "${stagec_bestfpr50_auc}" "${stagec_bestfpr50_fpr30}" "${stagec_bestfpr50_fpr50}" \
    "${baseline_auc}" "${baseline_fpr30}" "${baseline_fpr50}" \
    "${teacher_auc}" "${teacher_fpr30}" "${teacher_fpr50}" \
    >> "${summary_file}"
done

echo
echo "============================================================"
echo "Completed chunk ${CHUNK_ID}. Summary:"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "${summary_file}"
else
  cat "${summary_file}"
fi

echo
echo "Top configs by StageC selected FPR@30 (lower is better):"
python3 - "${summary_file}" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r.get("status") != "ok":
            continue
        try:
            rows.append((float(r["stagec_fpr30"]), r))
        except Exception:
            continue
rows.sort(key=lambda x: x[0])
for rank, (_, r) in enumerate(rows[:5], start=1):
    print(
        f"{rank:>2}. cfg={r['cfg_idx']} run={r['run_name']} "
        f"lrank={r['stageC_lambda_rank']} lreco={r['lambda_reco']} "
        f"lrd={r['stageC_lr_dual']} lrr={r['stageC_lr_reco']} "
        f"| stagec_auc={r['stagec_auc']} stagec_fpr30={r['stagec_fpr30']} stagec_fpr50={r['stagec_fpr50']}"
    )
PY

echo "Summary written to: ${summary_file}"
