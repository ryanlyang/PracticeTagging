#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(x: float) -> float:
    return 100.0 * float(x)


def make_dashboard(run_dir: Path, out_dir: Path) -> dict:
    results = np.load(run_dir / "results.npz")
    hlt = _load_json(run_dir / "hlt_stats.json")
    count = _load_json(run_dir / "constituent_count_summary.json")
    budget = _load_json(run_dir / "budget_summary_test.json")
    stages = _load_json(run_dir / "joint_stage_metrics.json")
    setup = _load_json(run_dir / "data_setup.json")

    auc_teacher = float(stages["test"]["auc_teacher"])
    auc_baseline = float(stages["test"]["auc_baseline"])
    auc_stage2 = float(stages["test"]["auc_stage2"])
    auc_joint = float(stages["test"]["auc_joint"])
    fpr30 = {
        "Teacher": float(stages["test"]["fpr30_teacher"]),
        "Baseline": float(stages["test"]["fpr30_baseline"]),
        "Stage2": float(stages["test"]["fpr30_stage2"]),
        "Joint": float(stages["test"]["fpr30_joint"]),
    }
    fpr50 = {
        "Teacher": float(stages["test"]["fpr50_teacher"]),
        "Baseline": float(stages["test"]["fpr50_baseline"]),
        "Stage2": float(stages["test"]["fpr50_stage2"]),
        "Joint": float(stages["test"]["fpr50_joint"]),
    }

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # ROC
    ax = axs[0, 0]
    ax.plot(results["tpr_teacher"], results["fpr_teacher"], label=f"Teacher ({auc_teacher:.3f})", lw=2)
    ax.plot(results["tpr_baseline"], results["fpr_baseline"], label=f"Baseline ({auc_baseline:.3f})", lw=2)
    ax.plot(results["tpr_stage2"], results["fpr_stage2"], label=f"Stage2 ({auc_stage2:.3f})", lw=2)
    ax.plot(results["tpr_joint"], results["fpr_joint"], label=f"Joint ({auc_joint:.3f})", lw=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("TPR")
    ax.set_ylabel("FPR (log)")
    ax.set_title("ROC")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # FPR bars
    ax = axs[0, 1]
    names = ["Teacher", "Baseline", "Stage2", "Joint"]
    x = np.arange(len(names))
    w = 0.38
    y30 = [_pct(fpr30[n]) for n in names]
    y50 = [_pct(fpr50[n]) for n in names]
    ax.bar(x - w / 2, y30, width=w, label="FPR@30%TPR")
    ax.bar(x + w / 2, y50, width=w, label="FPR@50%TPR")
    ax.set_xticks(x, names)
    ax.set_ylabel("FPR (%)")
    ax.set_title("Operating-Point Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # Count diagnostics
    ax = axs[1, 0]
    c_names = ["Offline", "HLT", "Reconstructed"]
    c_vals = [
        float(count["offline_count_mean"]),
        float(count["hlt_count_mean"]),
        float(count["reco_count_mean"]),
    ]
    ax.bar(c_names, c_vals, color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("Mean constituents / jet")
    ax.set_title("Constituent Counts")
    ax.grid(axis="y", alpha=0.25)
    txt = (
        f"MAE vs offline:\n"
        f"HLT = {count['hlt_count_mae_vs_offline']:.2f}\n"
        f"Reco = {count['reco_count_mae_vs_offline']:.2f}"
    )
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10)

    # Budget diagnostics
    ax = axs[1, 1]
    b_names = ["Merge", "Eff", "Total"]
    b_mae = [
        float(budget["merge_mae"]),
        float(budget["eff_mae"]),
        float(budget["total_mae"]),
    ]
    b_bias = [
        float(budget["merge_bias"]),
        float(budget["eff_bias"]),
        float(budget["total_bias"]),
    ]
    xx = np.arange(len(b_names))
    ax.bar(xx - 0.2, b_mae, width=0.4, label="MAE")
    ax.bar(xx + 0.2, b_bias, width=0.4, label="Bias")
    ax.set_xticks(xx, b_names)
    ax.set_ylabel("Constituent count units")
    ax.set_title("Budget Diagnostics (Test)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    mr = float(hlt["config"]["merge_radius"])
    rho = setup.get("rho", setup.get("added_target_scale", np.nan))
    rho = float(rho) if rho is not None else float("nan")
    pt_thr = float(hlt["config"]["pt_threshold_hlt"])
    fig.suptitle(
        f"Unmerger/Reconstructor Summary | merge_radius={mr:.3f}, rho={rho:.2f}, pt_threshold_hlt={pt_thr:.1f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "unmerger_dashboard.png", dpi=220)
    plt.close(fig)

    return {
        "auc_teacher": auc_teacher,
        "auc_baseline": auc_baseline,
        "auc_stage2": auc_stage2,
        "auc_joint": auc_joint,
        "fpr30_joint": float(fpr30["Joint"]),
        "fpr50_joint": float(fpr50["Joint"]),
        "merge_radius": mr,
        "rho": rho,
    }


def _short_name(name: str) -> str:
    prefixes = [
        "joint_100k_80c_stage2save_auc_norankc_",
        "joint_100k_80c_",
    ]
    out = name
    for p in prefixes:
        if out.startswith(p):
            out = out[len(p) :]
    return out


def make_tried_plot(all_runs_root: Path, out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in all_runs_root.rglob("joint_stage_metrics.json"):
        try:
            data = _load_json(p)
        except Exception:
            continue
        t = data.get("test", {})
        if not t:
            continue
        run = p.parent.name
        if "joint_100k_80c_stage2save_auc_norankc" not in run:
            continue
        if "nopriv" not in run and "lcons" not in run and "jetreg" not in run:
            continue
        row = {
            "run": run,
            "run_short": _short_name(run),
            "mode": data.get("variant", {}).get("mode", ""),
            "rho": data.get("variant", {}).get("rho", None),
            "auc_baseline": float(t.get("auc_baseline", np.nan)),
            "auc_joint": float(t.get("auc_joint", np.nan)),
            "fpr50_baseline": float(t.get("fpr50_baseline", np.nan)),
            "fpr50_joint": float(t.get("fpr50_joint", np.nan)),
            "fpr30_baseline": float(t.get("fpr30_baseline", np.nan)),
            "fpr30_joint": float(t.get("fpr30_joint", np.nan)),
        }
        row["auc_gain"] = row["auc_joint"] - row["auc_baseline"]
        row["fpr50_drop"] = row["fpr50_baseline"] - row["fpr50_joint"]
        row["fpr30_drop"] = row["fpr30_baseline"] - row["fpr30_joint"]
        rows.append(row)

    rows.sort(key=lambda r: (r["fpr50_joint"], -r["auc_joint"]))

    csv_path = out_dir / "tried_runs_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    if not rows:
        return rows

    n = min(10, len(rows))
    top = rows[:n]
    labels = [r["run_short"] for r in top]
    y = np.arange(n)
    fpr50_drop_pct = [100.0 * r["fpr50_drop"] for r in top]
    auc_gain_pts = [1000.0 * r["auc_gain"] for r in top]  # milli-AUC

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].barh(y, fpr50_drop_pct, color="#1f77b4")
    axs[0].set_yticks(y, labels)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("FPR@50 drop vs baseline (percentage points)")
    axs[0].set_title("Top Runs by Lowest Joint FPR@50")
    axs[0].grid(axis="x", alpha=0.25)

    axs[1].barh(y, auc_gain_pts, color="#2ca02c")
    axs[1].set_yticks(y, labels)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("AUC gain vs baseline (milli-AUC)")
    axs[1].set_title("AUC Gains")
    axs[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "things_tried_comparison.png", dpi=220)
    plt.close(fig)
    return rows


def write_reply_draft(run_dir: Path, out_dir: Path, summary: dict) -> None:
    setup = _load_json(run_dir / "data_setup.json")
    hlt = _load_json(run_dir / "hlt_stats.json")
    count = _load_json(run_dir / "constituent_count_summary.json")
    budget = _load_json(run_dir / "budget_summary_test.json")

    lines = []
    lines.append("Hi, thanks for following up and sorry you missed the meeting.")
    lines.append("")
    lines.append("Below is a quick summary of the unmerging/reconstruction model and key results.")
    lines.append("")
    lines.append("Model summary:")
    lines.append("- Constituent-level transformer reconstructor with relative-position attention bias in (deta, dphi, dR).")
    lines.append("- Predicts token actions (keep/unsmear/split/reassign), split children, generated recovery tokens, and jet-level budgets.")
    lines.append("- Trained with set-level matching + jet-physics consistency + budget/count constraints; then jointly optimized with the dual-view top tagger.")
    lines.append("")
    lines.append("Main run settings used in the shared slides/results:")
    lines.append(f"- merge_radius = {hlt['config']['merge_radius']}")
    lines.append(f"- pt_threshold_hlt = {hlt['config']['pt_threshold_hlt']} GeV")
    lines.append(
        f"- efficiency model enabled (eta/pt/density dependent), smearing enabled, local reassignment enabled"
    )
    rho_val = setup.get("rho", setup.get("added_target_scale", "n/a"))
    lines.append(f"- non-privileged rho split/scale target with rho = {rho_val}")
    lines.append("")
    lines.append("Performance snapshot (test):")
    lines.append(
        f"- AUC: baseline={summary['auc_baseline']:.4f}, stage2={summary['auc_stage2']:.4f}, joint={summary['auc_joint']:.4f}, teacher={summary['auc_teacher']:.4f}"
    )
    lines.append(
        f"- FPR@30 (joint): {100.0*summary['fpr30_joint']:.3f}% | FPR@50 (joint): {100.0*summary['fpr50_joint']:.3f}%"
    )
    lines.append(
        f"- Constituent count MAE vs offline: HLT={count['hlt_count_mae_vs_offline']:.3f}, reconstructed={count['reco_count_mae_vs_offline']:.3f}"
    )
    lines.append(
        f"- Budget MAE (test): merge={budget['merge_mae']:.3f}, eff={budget['eff_mae']:.3f}, total={budget['total_mae']:.3f}"
    )
    lines.append("")
    lines.append("Attached plots:")
    lines.append("- unmerger_dashboard.png")
    lines.append("- things_tried_comparison.png")
    lines.append("- tried_runs_summary.csv")
    lines.append("")
    lines.append("Happy to share the full run config/checkpoints and walk through any specific ablation in detail.")
    (out_dir / "reply_draft.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=Path(
            "/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data/download_checkpoints/"
            "joint_100k_80c_stage2save_auc_norankc_nopriv_rhosplit_splitagain_rho090_noflags"
        ),
    )
    ap.add_argument(
        "--all_runs_root",
        type=Path,
        default=Path("/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data/download_checkpoints"),
    )
    args = ap.parse_args()

    out_dir = args.run_dir / "researcher_followup_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = make_dashboard(args.run_dir, out_dir)
    make_tried_plot(args.all_runs_root, out_dir)
    write_reply_draft(args.run_dir, out_dir, summary)
    print(f"Saved follow-up pack to: {out_dir}")


if __name__ == "__main__":
    main()
