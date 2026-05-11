#!/usr/bin/env bash
#SBATCH --job-name=jt12chk
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/check_joint12_fused_targets_split_150k150k300k_%j.out
#SBATCH --error=offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k/check_joint12_fused_targets_split_150k150k300k_%j.err

set -euo pipefail

mkdir -p offline_reconstructor_logs/reco_teacher_joint_fusion_6model_150k75k150k

FUSED_TARGETS_NPZ="${FUSED_TARGETS_NPZ:-checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/fused_targets_joint12_weighted_150k150k300k_split/fused_targets_train_val_test.npz}"

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python - <<'PY'
import numpy as np
import os
from pathlib import Path

npz_path = Path(os.environ["FUSED_TARGETS_NPZ"]).expanduser().resolve()
if not npz_path.exists():
    raise FileNotFoundError(f"Missing fused targets NPZ: {npz_path}")

arr = np.load(npz_path)
if "idx_fit" not in arr or "idx_ref" not in arr:
    raise KeyError(
        f"Missing idx_fit/idx_ref in {npz_path}. Keys: {sorted(arr.files)}"
    )

idx_fit = np.asarray(arr["idx_fit"], dtype=np.int64).reshape(-1)
idx_ref = np.asarray(arr["idx_ref"], dtype=np.int64).reshape(-1)

eq = np.array_equal(idx_fit, idx_ref)
ov = len(set(idx_fit.tolist()).intersection(set(idx_ref.tolist())))

print("============================================================")
print("Check fused target split integrity")
print("============================================================")
print(f"NPZ:      {npz_path}")
print(f"n_fit:    {idx_fit.size}")
print(f"n_ref:    {idx_ref.size}")
print(f"equal:    {eq}")
print(f"overlap:  {ov}")

if eq or ov > 0:
    raise SystemExit(
        "ERROR: idx_fit/idx_ref are not disjoint. Do not use these targets for train/val split KD."
    )

print("PASS: idx_fit and idx_ref are disjoint.")
PY
