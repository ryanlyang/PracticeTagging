#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(run_dir: Path) -> dict:
    count = load_json(run_dir / "constituent_count_summary.json")
    budget = load_json(run_dir / "budget_summary_test.json")
    hlt = load_json(run_dir / "hlt_stats.json")
    stage = load_json(run_dir / "joint_stage_metrics.json")
    arr = np.load(run_dir / "results.npz")

    hlt_resp = np.array(arr["jet_response_hlt_mean"], dtype=np.float64)
    reco_resp = np.array(arr["jet_response_corrected_mean"], dtype=np.float64)
    hlt_reso = np.array(arr["jet_response_hlt_std"], dtype=np.float64)
    reco_reso = np.array(arr["jet_response_corrected_std"], dtype=np.float64)

    out = {
        "run_name": run_dir.name,
        "merge_radius": float(hlt["config"]["merge_radius"]),
        "pt_threshold_hlt": float(hlt["config"]["pt_threshold_hlt"]),
        "n_jets": int(hlt["stats"]["n_jets"]),
        "auc_baseline": float(stage["test"]["auc_baseline"]),
        "auc_joint": float(stage["test"]["auc_joint"]),
        "count_offline_mean": float(count["offline_count_mean"]),
        "count_hlt_mean": float(count["hlt_count_mean"]),
        "count_reco_mean": float(count["reco_count_mean"]),
        "count_mae_hlt": float(count["hlt_count_mae_vs_offline"]),
        "count_mae_reco": float(count["reco_count_mae_vs_offline"]),
        "count_mae_reduction_pct": float(
            100.0
            * (1.0 - float(count["reco_count_mae_vs_offline"]) / max(float(count["hlt_count_mae_vs_offline"]), 1e-12))
        ),
        "lost_merge_mean_per_jet": float(count["lost_merge_mean_per_jet"]),
        "created_merge_mean_per_jet": float(count["created_merge_mean_per_jet"]),
        "needed_add_mean_per_jet": float(count["needed_add_mean_per_jet"]),
        "created_total_mean_per_jet": float(count["created_total_mean_per_jet"]),
        "merge_recovery_ratio": float(
            float(count["created_merge_mean_per_jet"]) / max(float(count["lost_merge_mean_per_jet"]), 1e-12)
        ),
        "add_gap_fill_ratio": float(
            float(count["created_total_mean_per_jet"]) / max(float(count["needed_add_mean_per_jet"]), 1e-12)
        ),
        "budget_mae_merge": float(budget["merge_mae"]),
        "budget_mae_eff": float(budget["eff_mae"]),
        "budget_mae_total": float(budget["total_mae"]),
        "budget_bias_merge": float(budget["merge_bias"]),
        "budget_bias_eff": float(budget["eff_bias"]),
        "budget_bias_total": float(budget["total_bias"]),
        "jet_response_hlt_mean": float(np.mean(hlt_resp)),
        "jet_response_reco_mean": float(np.mean(reco_resp)),
        "jet_resolution_hlt_mean": float(np.mean(hlt_reso)),
        "jet_resolution_reco_mean": float(np.mean(reco_reso)),
    }
    return out


def make_main_plot(run_dir: Path, out_dir: Path, s: dict) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Counts
    ax = axs[0, 0]
    names = ["Offline", "HLT", "Reconstructed"]
    vals = [s["count_offline_mean"], s["count_hlt_mean"], s["count_reco_mean"]]
    ax.bar(names, vals, color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_title("Mean Constituent Count / Jet")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    # Unmerge recovery
    ax = axs[0, 1]
    labels = ["Lost merge", "Created merge", "Needed add", "Created total"]
    vals = [
        s["lost_merge_mean_per_jet"],
        s["created_merge_mean_per_jet"],
        s["needed_add_mean_per_jet"],
        s["created_total_mean_per_jet"],
    ]
    ax.bar(labels, vals, color=["#1f77b4", "#2ca02c", "#7f7f7f", "#17becf"])
    ax.set_title("Unmerging Recovery Statistics")
    ax.set_ylabel("Constituents / jet")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=15)
    txt = (
        f"merge recovery = {s['merge_recovery_ratio']:.3f}\n"
        f"gap fill ratio = {s['add_gap_fill_ratio']:.3f}"
    )
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10)

    # Count MAE + budget MAE
    ax = axs[1, 0]
    x = np.arange(5)
    lbl = ["Count MAE HLT", "Count MAE Reco", "Budget MAE Merge", "Budget MAE Eff", "Budget MAE Total"]
    v = [
        s["count_mae_hlt"],
        s["count_mae_reco"],
        s["budget_mae_merge"],
        s["budget_mae_eff"],
        s["budget_mae_total"],
    ]
    ax.bar(x, v, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"])
    ax.set_xticks(x, lbl, rotation=20, ha="right")
    ax.set_ylabel("Constituent count units")
    ax.set_title("Error Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.95,
        f"Count MAE reduction: {s['count_mae_reduction_pct']:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    # Jet response/resolution
    ax = axs[1, 1]
    names = ["Mean response", "Mean resolution"]
    hlt_vals = [s["jet_response_hlt_mean"], s["jet_resolution_hlt_mean"]]
    reco_vals = [s["jet_response_reco_mean"], s["jet_resolution_reco_mean"]]
    xx = np.arange(len(names))
    w = 0.35
    ax.bar(xx - w / 2, hlt_vals, width=w, label="HLT")
    ax.bar(xx + w / 2, reco_vals, width=w, label="Reconstructed")
    ax.set_xticks(xx, names)
    ax.set_title("Jet-Level Accuracy vs Offline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle(
        f"Reconstructor Accuracy Summary | {s['run_name']} | merge_radius={s['merge_radius']:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "reconstructor_accuracy_summary.png", dpi=220)
    plt.close(fig)


def make_response_plot(run_dir: Path, out_dir: Path) -> None:
    arr = np.load(run_dir / "results.npz")
    x = 0.5 * (arr["jet_response_pt_low"] + arr["jet_response_pt_high"])
    y_hlt = arr["jet_response_hlt_mean"]
    y_rec = arr["jet_response_corrected_mean"]
    s_hlt = arr["jet_response_hlt_std"]
    s_rec = arr["jet_response_corrected_std"]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(x, y_hlt, "o-", label="HLT")
    axs[0].plot(x, y_rec, "s-", label="Reconstructed")
    axs[0].set_title("Jet Response vs Truth pT Bin")
    axs[0].set_xlabel("Truth jet pT bin center")
    axs[0].set_ylabel("Mean pT_reco / pT_offline")
    axs[0].grid(alpha=0.25)
    axs[0].legend(frameon=False)

    axs[1].plot(x, s_hlt, "o-", label="HLT")
    axs[1].plot(x, s_rec, "s-", label="Reconstructed")
    axs[1].set_title("Jet Resolution vs Truth pT Bin")
    axs[1].set_xlabel("Truth jet pT bin center")
    axs[1].set_ylabel("Width of response")
    axs[1].grid(alpha=0.25)
    axs[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "reconstructor_response_resolution_by_pt.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.run_dir / "reconstructor_focus_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args.run_dir)
    (out_dir / "reconstructor_summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    make_main_plot(args.run_dir, out_dir, summary)
    make_response_plot(args.run_dir, out_dir)
    print(f"Saved reconstructor-focused pack to: {out_dir}")


if __name__ == "__main__":
    main()
