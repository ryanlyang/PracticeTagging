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
    ap.add_argument(
        "--mitigated",
        default="dual_flag_kd",
        help="Which curve to treat as the mitigated model (suffix in results.npz keys).",
    )
    ap.add_argument(
        "--log_fpr",
        action="store_true",
        help="Plot FPR (y-axis) on log scale.",
    )
    ap.add_argument(
        "--fpr_min",
        type=float,
        default=1e-4,
        help="Minimum FPR for log y-axis (if data allows).",
    )
    ap.add_argument(
        "--tpr_points",
        type=str,
        default="0.5,0.3",
        help="Comma-separated TPR working points for table (e.g. 0.5,0.3).",
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
    mit_key = args.mitigated
    fpr_m = need(f"fpr_{mit_key}")
    tpr_m = need(f"tpr_{mit_key}")

    # Required AUCs
    auc_t = float(need("auc_teacher"))
    auc_b = float(need("auc_baseline"))
    auc_m = float(need(f"auc_{mit_key}"))

    out_path = Path(args.out) if args.out else npz_path.with_name(
        "results_teacher_baseline_dualview_mf_kd.png"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_t:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"HLT Baseline (AUC={auc_b:.3f})", color="steelblue", linewidth=2)
    plt.plot(
        tpr_m,
        fpr_m,
        "-.",
        label=f"{mit_key} (AUC={auc_m:.3f})",
        color="darkslateblue",
        linewidth=2,
    )
    if args.log_fpr:
        # Choose y-min: 1e-4 if data reaches it, otherwise clamp to min observed positive fpr.
        min_pos = min(np.min(fpr_t[fpr_t > 0]), np.min(fpr_b[fpr_b > 0]), np.min(fpr_m[fpr_m > 0]))
        y_min = args.fpr_min if min_pos <= args.fpr_min else min_pos
        plt.yscale("log")
        plt.ylim(y_min, 1.0)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")

    # Table: FPR at fixed TPRs
    tpr_points = [float(x) for x in args.tpr_points.split(",") if x.strip() != ""]

    def fpr_at_tpr(fpr, tpr, target):
        # Ensure monotonic by sorting on tpr
        order = np.argsort(tpr)
        t = tpr[order]
        f = fpr[order]
        if target < t[0] or target > t[-1]:
            return float("nan")
        return float(np.interp(target, t, f))

    print("\nFPR at fixed TPRs (percent)")
    header = ["TPR (%)", "HLT (baseline) %", f"{mit_key} %", "Offline (teacher) %"]
    print("{:>10} | {:>18} | {:>12} | {:>20}".format(*header))
    print("-" * 75)
    for tp in tpr_points:
        f_b = fpr_at_tpr(fpr_b, tpr_b, tp)
        f_m = fpr_at_tpr(fpr_m, tpr_m, tp)
        f_t = fpr_at_tpr(fpr_t, tpr_t, tp)
        print(
            "{:>10.1f} | {:>18.3f} | {:>12.3f} | {:>20.3f}".format(
                tp * 100.0, f_b * 100.0, f_m * 100.0, f_t * 100.0
            )
        )


if __name__ == "__main__":
    main()
