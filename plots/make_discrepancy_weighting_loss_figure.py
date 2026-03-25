#!/usr/bin/env python3
"""Generate a visual explainer for discrepancy weighting and weighted losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#1f2937",
    title_size: int = 12,
    body_size: int = 10,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + h - 0.03,
        title,
        fontsize=title_size,
        fontweight="bold",
        color="#0f172a",
        va="top",
    )
    ax.text(
        x + 0.012,
        y + h - 0.075,
        body,
        fontsize=body_size,
        color="#111827",
        va="top",
        linespacing=1.32,
    )


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#374151",
    lw: float = 1.8,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def make_figure() -> plt.Figure:
    fig = plt.figure(figsize=(15.2, 9.4), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.text(
        0.03,
        0.965,
        "Discrepancy Weighting -> Weighted Losses (Unmerge-Only Stage2save Variant)",
        fontsize=17,
        fontweight="bold",
        color="#0b1220",
        va="top",
    )
    ax.text(
        0.03,
        0.938,
        "Core idea: use teacher-vs-HLT disagreement to upweight hard jets, then apply weighted means in Stage A/B/C.",
        fontsize=10.6,
        color="#334155",
        va="top",
    )

    draw_box(
        ax,
        0.03,
        0.69,
        0.28,
        0.215,
        "1) Build Score Inputs",
        "Train teacher on offline view and baseline on HLT view.\n"
        "Collect per-jet probabilities on train + val:\n"
        "  p_teacher, p_hlt, label y\n"
        "Val positives define operating thresholds at target TPR.",
        facecolor="#e0f2fe",
    )

    draw_box(
        ax,
        0.35,
        0.69,
        0.28,
        0.215,
        "2) Apply Gates (default: ON)",
        "gate = hard_teacher_correct\n"
        "     * sigmoid((p_teacher_true - conf_min)/tau_c)\n"
        "     * sigmoid((p_teacher_true - p_hlt_true)/tau_c)\n"
        "If a gate is low, discrepancy signal is suppressed.",
        facecolor="#dcfce7",
    )

    draw_box(
        ax,
        0.67,
        0.69,
        0.30,
        0.215,
        "3) Discrepancy Risk r",
        "smooth_delta mode (this runner):\n"
        "  r_neg = max(p_hlt - p_teacher, 0) * 1[y=0] * gate\n"
        "  r = r_neg (+ optional positive term if include_pos)\n\n"
        "tail_disagreement mode (alt): uses sigmoid tails around\n"
        "val thresholds (t_hlt, t_off) with temperature tau.",
        facecolor="#fef3c7",
    )

    draw_arrow(ax, (0.31, 0.795), (0.35, 0.795))
    draw_arrow(ax, (0.63, 0.795), (0.67, 0.795))

    draw_box(
        ax,
        0.245,
        0.455,
        0.51,
        0.155,
        "4) Convert r -> Per-Jet Weight w",
        "w = clip(1 + lambda_disc * r, 1, max_mult)\n"
        "if mean-normalize enabled:  w <- w / mean(w)\n"
        "This preserves global scale while redistributing emphasis toward high-discrepancy jets.",
        facecolor="#ede9fe",
        title_size=13,
        body_size=11,
    )

    draw_arrow(ax, (0.49, 0.69), (0.49, 0.61))
    draw_arrow(ax, (0.82, 0.69), (0.70, 0.61))
    draw_arrow(ax, (0.17, 0.69), (0.30, 0.61))

    draw_box(
        ax,
        0.03,
        0.12,
        0.29,
        0.265,
        "5A) Stage A (Reconstructor)",
        "Use sample_weight_reco for reconstruction components:\n"
        "  L_reco^w = sum_i w_reco_i * L_reco_i / sum_i w_reco_i\n"
        "Selection metric in Stage A uses weighted val total.\n"
        "Purpose: focus correction capacity on jets where HLT\n"
        "looks overly signal-like vs teacher (mostly negatives).",
        facecolor="#fee2e2",
    )

    draw_box(
        ax,
        0.355,
        0.12,
        0.29,
        0.265,
        "5B) Stage B (Frozen Reco + Dual BCE)",
        "Use sample_weight_cls in classifier BCE:\n"
        "  L_cls^w = sum_i w_cls_i * BCE_i / sum_i w_cls_i\n"
        "Reco term disabled here (lambda_reco = 0 in Stage B).\n"
        "Validation AUC/FPR can be weighted with sample_weight_cls\n"
        "for checkpoint selection.",
        facecolor="#ffedd5",
    )

    draw_box(
        ax,
        0.68,
        0.12,
        0.29,
        0.265,
        "5C) Stage C (Joint)",
        "Total objective:\n"
        "  L = L_cls (+weighted if enabled)\n"
        "    + lambda_reco * L_reco (+weighted if enabled)\n"
        "    + lambda_cons * L_cons (+ optional delta term)\n"
        "In this runner: weighted reco ON; weighted cls OFF by default\n"
        "unless disc_apply_cls_stagec is set.",
        facecolor="#dcfce7",
    )

    draw_arrow(ax, (0.49, 0.455), (0.175, 0.385))
    draw_arrow(ax, (0.49, 0.455), (0.50, 0.385))
    draw_arrow(ax, (0.49, 0.455), (0.825, 0.385))

    draw_box(
        ax,
        0.03,
        0.02,
        0.94,
        0.075,
        "Runner Defaults in run_offline_reconstructor...discweighted_50k10k100k_80c.sh",
        "mode=smooth_delta | reco lambda/max=6.0/8.0 | cls lambda/max=2.0/3.0 | target_tpr=0.50 | tau=0.05 | "
        "teacher_conf_min=0.65 | correctness_tau=0.05 | mean-normalization ON | include_pos OFF | "
        "all three teacher gates enabled.",
        facecolor="#e2e8f0",
        title_size=10,
        body_size=9,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create discrepancy-weighted loss explainer figure.")
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("images/discrepancy_weighting_loss_explainer.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=Path("images/discrepancy_weighting_loss_explainer.svg"),
        help="Output SVG path.",
    )
    args = parser.parse_args()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.out_png}")
    print(f"Wrote: {args.out_svg}")


if __name__ == "__main__":
    main()
