#!/usr/bin/env bash
#SBATCH --job-name=an55rollpc
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze55_roll_precomp_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/analyze55_roll_precomp_%j.err

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

TARGET_TPR="${TARGET_TPR:-0.50}"
ANCHOR_MODEL="${ANCHOR_MODEL:-joint_delta}"
CANDIDATE_MODELS="${CANDIDATE_MODELS:-}"

VAL_POOL_SIZE="${VAL_POOL_SIZE:-1000000}"
VAL_POOL_OFFSET="${VAL_POOL_OFFSET:-0}"
N_CHUNKS="${N_CHUNKS:-5}"

CANDIDATE_TOPK_FIT="${CANDIDATE_TOPK_FIT:-55}"
MAX_STEPS="${MAX_STEPS:-20}"
PATIENCE="${PATIENCE:-5}"
W_STEP="${W_STEP:-0.01}"
CALIBRATION="${CALIBRATION:-raw}"
MIN_FIT_GAIN="${MIN_FIT_GAIN:-1e-6}"
MIN_CAL_GAIN="${MIN_CAL_GAIN:-1e-6}"
MIN_CUM_CAL_GAIN="${MIN_CUM_CAL_GAIN:-1e-6}"
TOPK_DIAGNOSTIC="${TOPK_DIAGNOSTIC:-10}"
SEED="${SEED:-0}"

PRECOMP_DIR="${PRECOMP_DIR:-}"
PRECOMP_SCORES_NPZ="${PRECOMP_SCORES_NPZ:-}"
PRECOMP_MANIFEST_JSON="${PRECOMP_MANIFEST_JSON:-}"

OUT_DIR="${OUT_DIR:-}"
REPORT_JSON="${REPORT_JSON:-}"

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

if [[ -z "${PRECOMP_DIR}" ]]; then
  PRECOMP_DIR="$(dirname "${FUSION_JSON}")/precomputed_devpool_1000000"
fi
if [[ -z "${PRECOMP_SCORES_NPZ}" ]]; then
  PRECOMP_SCORES_NPZ="${PRECOMP_DIR}/precomputed_scores_devpool.npz"
fi
if [[ -z "${PRECOMP_MANIFEST_JSON}" ]]; then
  PRECOMP_MANIFEST_JSON="${PRECOMP_DIR}/precomputed_scores_manifest.json"
fi

if [[ ! -f "${PRECOMP_SCORES_NPZ}" ]]; then
  echo "ERROR: PRECOMP_SCORES_NPZ not found: ${PRECOMP_SCORES_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${PRECOMP_MANIFEST_JSON}" ]]; then
  echo "ERROR: PRECOMP_MANIFEST_JSON not found: ${PRECOMP_MANIFEST_JSON}" >&2
  exit 1
fi

CMD=(
  python analyze_hlt_joint31_rolling_chunk_greedy.py
  --fusion_json "${FUSION_JSON}"
  --precomputed_scores_npz "${PRECOMP_SCORES_NPZ}"
  --precomputed_manifest_json "${PRECOMP_MANIFEST_JSON}"
  --target_tpr "${TARGET_TPR}"
  --anchor_model "${ANCHOR_MODEL}"
  --candidate_topk_fit "${CANDIDATE_TOPK_FIT}"
  --val_pool_size "${VAL_POOL_SIZE}"
  --val_pool_offset "${VAL_POOL_OFFSET}"
  --n_chunks "${N_CHUNKS}"
  --max_steps "${MAX_STEPS}"
  --patience "${PATIENCE}"
  --w_step "${W_STEP}"
  --calibration "${CALIBRATION}"
  --min_fit_gain "${MIN_FIT_GAIN}"
  --min_cal_gain "${MIN_CAL_GAIN}"
  --min_cum_cal_gain "${MIN_CUM_CAL_GAIN}"
  --topk_diagnostic "${TOPK_DIAGNOSTIC}"
  --seed "${SEED}"
)

if [[ -n "${CANDIDATE_MODELS}" ]]; then
  CMD+=(--candidate_models "${CANDIDATE_MODELS}")
fi
if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "${OUT_DIR}")
fi
if [[ -n "${REPORT_JSON}" ]]; then
  CMD+=(--report_json "${REPORT_JSON}")
fi

echo "============================================================"
echo "55-Model Rolling-Chunk Greedy Fusion (Precomputed Dev Pool)"
echo "Fusion json:         ${FUSION_JSON}"
echo "Precomputed npz:     ${PRECOMP_SCORES_NPZ}"
echo "Precomputed manifest:${PRECOMP_MANIFEST_JSON}"
echo "Target TPR:          ${TARGET_TPR}"
echo "Anchor:              ${ANCHOR_MODEL}"
echo "Val pool size/offset:${VAL_POOL_SIZE}/${VAL_POOL_OFFSET}"
echo "Chunks:              ${N_CHUNKS}"
echo "Greedy:              topk=${CANDIDATE_TOPK_FIT}, max_steps=${MAX_STEPS}, w_step=${W_STEP}"
echo "Acceptance gains:    fit>=${MIN_FIT_GAIN}, cal>=${MIN_CAL_GAIN}, cum>=${MIN_CUM_CAL_GAIN}"
echo "Calibration:         ${CALIBRATION}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
