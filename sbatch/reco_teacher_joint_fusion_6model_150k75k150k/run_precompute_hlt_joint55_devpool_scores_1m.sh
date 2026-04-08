#!/usr/bin/env bash
#SBATCH --job-name=pre55_1m
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/precompute55_1m_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/precompute55_1m_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

FUSION_JSON_DEFAULT_CHECKPOINTS="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"
FUSION_JSON_DEFAULT_DOWNLOAD_A="download_checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_joint_delta/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"
FUSION_JSON_DEFAULT_DOWNLOAD_B="download_checkpoints/model2_joint_delta005_150k75k150k_seed0/fusion_hlt_joint31_all_stackers.json"

if [[ -z "${FUSION_JSON:-}" ]]; then
  if [[ -f "${FUSION_JSON_DEFAULT_CHECKPOINTS}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_CHECKPOINTS}"
  elif [[ -f "${FUSION_JSON_DEFAULT_DOWNLOAD_A}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_DOWNLOAD_A}"
  elif [[ -f "${FUSION_JSON_DEFAULT_DOWNLOAD_B}" ]]; then
    FUSION_JSON="${FUSION_JSON_DEFAULT_DOWNLOAD_B}"
  else
    FUSION_JSON="${FUSION_JSON_DEFAULT_CHECKPOINTS}"
  fi
fi

TRAIN_PATH="${TRAIN_PATH:-./data}"
DEV_OFFSET_JETS="${DEV_OFFSET_JETS:-375000}"
DEV_N_JETS="${DEV_N_JETS:-1000000}"
MAX_CONSTITS="${MAX_CONSTITS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"
HLT_SEED="${HLT_SEED:--1}"
CORRECTED_WEIGHT_FLOOR_JOINT="${CORRECTED_WEIGHT_FLOOR_JOINT:-1e-4}"
CORRECTED_WEIGHT_FLOOR_STAGEA="${CORRECTED_WEIGHT_FLOOR_STAGEA:-0.03}"

OUT_DIR="${OUT_DIR:-}"
SCORES_NPZ="${SCORES_NPZ:-}"
MANIFEST_JSON="${MANIFEST_JSON:-}"
MONITOR_CSV="${MONITOR_CSV:-}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "${FUSION_JSON}" ]]; then
  echo "ERROR: FUSION_JSON not found: ${FUSION_JSON}" >&2
  exit 1
fi

# Extend base 31-model fusion json to 55 models:
# 31 base + 15 prior additions + 9 additional high-value runs.
EXTRA_MODELS=(
  # Prior 15 additions
  "model2_joint_delta005_stagec_prog_unfreeze_splitctx_heads_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_jetlatent_set2set_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_unselected0_localr0_stagec_prog_wlocal010_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_hybridcand_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_splitcap120_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_linear_unsmear_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_hungarian_tokenaux_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_unselected0_localr0_stagec_prog_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_factorized_edit_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_eratioon_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_physon_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_splitmass_softmax_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_actionentropy_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_stagea_taskaux_150k75k150k_seed0"
  "model2_joint_delta005_fulltrain_prog_unfreeze_splitk3_softgate_150k75k150k_seed0"
  # Additional 9 high-value additions (includes unsmearcap_trust)
  "model2_joint_delta005_stagec_prog_unfreeze_150k75k150k_seed0"
  "model2_joint_chamfer_budget_local_ptratio_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_ot_sinkhorn_nophys_noratio_150k75k150k_seed0"
  "model2_joint_hungarian_nophys_noratio_150k75k150k_seed0"
  "model2_joint_chamfer_phys0_eratio035_unselected0_localr0_150k75k150k_seed0"
  "model2_joint_ot_chamfer_sinkhorn_nophys_noratio_150k75k150k_seed0"
  "model2_joint_sinkhorn_150k75k150k_seed0"
  "model2_joint_chamfer_budget_sparse_150k75k150k_seed0"
  "model2_joint_delta000_unsmearcap_trust_150k75k150k_seed0"
)

EXTENDED_FUSION_JSON="${EXTENDED_FUSION_JSON:-${FUSION_JSON%.json}_rolling55.json}"
export EXTRA_MODELS_CSV="$(IFS=,; echo "${EXTRA_MODELS[*]}")"
python3 - "${FUSION_JSON}" "${EXTENDED_FUSION_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
fusion = json.loads(src.read_text())

extra = [x.strip() for x in os.environ.get("EXTRA_MODELS_CSV", "").split(",") if x.strip()]
run_dirs = fusion.get("run_dirs", {})
if not isinstance(run_dirs, dict):
    run_dirs = {}
fusion["run_dirs"] = run_dirs
score_files = run_dirs.get("score_files", {})
if not isinstance(score_files, dict):
    score_files = {}
run_dirs["score_files"] = score_files

models_order = fusion.get("models_order", [])
if not isinstance(models_order, list):
    models_order = []

base_ckpt = Path("checkpoints/reco_teacher_joint_fusion_6model_150k75k150k").expanduser().resolve()
suffix = "_150k75k150k_seed0"

def _fallback_path(mid: str) -> Path:
    if mid.endswith(suffix):
        parent = mid[: -len(suffix)]
    else:
        parent = mid
    return (base_ckpt / parent / mid / "fusion_scores_val_test.npz").resolve()

def _resolve_score_path(mid: str) -> Path:
    fb = _fallback_path(mid)
    if fb.exists():
        return fb
    if base_ckpt.exists():
        hits = [p.resolve() for p in base_ckpt.glob(f"**/{mid}/fusion_scores_val_test.npz")]
        if hits:
            hits = sorted(set(hits), key=lambda p: (len(str(p)), str(p)))
            return hits[0]
    return fb

missing = []
for mid in extra:
    p = _resolve_score_path(mid)
    score_files[mid] = str(p)
    if not p.exists():
        missing.append(mid)
    if mid not in models_order:
        models_order.append(mid)

fusion["models_order"] = models_order
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(fusion, indent=2))
print(f"Wrote extended fusion json: {dst}")
print(f"Total models_order: {len(models_order)}")
if missing:
    print("WARNING: unresolved score paths for models:")
    for m in missing:
        print(f"  - {m}")
PY
FUSION_JSON="${EXTENDED_FUSION_JSON}"

CMD=(
  python precompute_hlt_joint55_devpool_scores.py
  --fusion_json "${FUSION_JSON}"
  --train_path "${TRAIN_PATH}"
  --dev_offset_jets "${DEV_OFFSET_JETS}"
  --dev_n_jets "${DEV_N_JETS}"
  --max_constits "${MAX_CONSTITS}"
  --hlt_seed "${HLT_SEED}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --corrected_weight_floor_joint "${CORRECTED_WEIGHT_FLOOR_JOINT}"
  --corrected_weight_floor_stagea "${CORRECTED_WEIGHT_FLOOR_STAGEA}"
)

if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${SCORES_NPZ}" ]]; then
  CMD+=(--scores_npz "${SCORES_NPZ}")
fi
if [[ -n "${MANIFEST_JSON}" ]]; then
  CMD+=(--manifest_json "${MANIFEST_JSON}")
fi
if [[ -n "${MONITOR_CSV}" ]]; then
  CMD+=(--monitor_csv "${MONITOR_CSV}")
fi

echo "============================================================"
echo "Precompute 55-Model Dev-Pool Scores"
echo "Fusion json:      ${FUSION_JSON}"
echo "Train path:       ${TRAIN_PATH}"
echo "Dev offset/n:     ${DEV_OFFSET_JETS}/${DEV_N_JETS}"
echo "Max constits:     ${MAX_CONSTITS}"
echo "Batch/device:     ${BATCH_SIZE}/${DEVICE}"
echo "Weight floors:    joint=${CORRECTED_WEIGHT_FLOOR_JOINT}, stageA=${CORRECTED_WEIGHT_FLOOR_STAGEA}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
