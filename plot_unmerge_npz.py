#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to results.npz")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: alongside npz, name results_teacher_baseline_dualview_mf_kd.png)",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    keys = set(data.files)

    def need(k: str) -> np.ndarray:
        if k not in keys:
            raise KeyError(f"Missing key '{k}' in {npz_path}. Keys: {sorted(keys)}")
        return data[k]

    # Required curves
    fpr_t = need("fpr_teacher")
    tpr_t = need("tpr_teacher")
    fpr_b = need("fpr_baseline")
    tpr_b = need("tpr_baseline")
    fpr_dvf_k = need("fpr_dual_flag_kd")
    tpr_dvf_k = need("tpr_dual_flag_kd")

    # Required AUCs
    auc_t = float(need("auc_teacher"))
    auc_b = float(need("auc_baseline"))
    auc_dvf_k = float(need("auc_dual_flag_kd"))

    out_path = Path(args.out) if args.out else npz_path.with_name(
        "results_teacher_baseline_dualview_mf_kd.png"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_t:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"HLT Baseline (AUC={auc_b:.3f})", color="steelblue", linewidth=2)
    plt.plot(
        tpr_dvf_k,
        fpr_dvf_k,
        "-.",
        label=f"DualView+MF+KD (AUC={auc_dvf_k:.3f})",
        color="darkslateblue",
        linewidth=2,
    )
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
