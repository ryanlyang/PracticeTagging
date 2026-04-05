#!/usr/bin/env python3
"""Create bolder, low-text soft-vs-hard conceptual slide graphics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def setup(title: str):
    fig = plt.figure(figsize=(13.6, 7.6), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7fafc")
    ax.text(0.03, 0.95, title, fontsize=26, fontweight="bold", color="#0b1220", va="top")
    return fig, ax


def arr(ax, s, e, c="#111827", lw=3.0, ms=18):
    ax.add_patch(
        FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=c, connectionstyle="arc3,rad=0")
    )


def panel(ax, x, y, w, h, fc, ec="#0f172a"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(p)


def slide1():
    fig, ax = setup("Hard View: Predict Offline From HLT, But Output Only One Choice")
    panel(ax, 0.04, 0.16, 0.27, 0.68, "#dbeafe")
    panel(ax, 0.365, 0.16, 0.27, 0.68, "#fee2e2")
    panel(ax, 0.69, 0.16, 0.27, 0.68, "#ffedd5")

    ax.text(0.075, 0.80, "OFFLINE", fontsize=18, fontweight="bold", color="#1e3a8a")
    ax.text(0.415, 0.80, "HLT (INPUT)", fontsize=18, fontweight="bold", color="#991b1b")
    ax.text(0.73, 0.80, "PREDICT OFFLINE", fontsize=18, fontweight="bold", color="#9a3412")

    # offline rich tokens
    for cx, cy in [(0.11, 0.62), (0.19, 0.60), (0.26, 0.64), (0.13, 0.48), (0.22, 0.47), (0.27, 0.40)]:
        ax.add_patch(Circle((cx, cy), 0.025, facecolor="#60a5fa", edgecolor="#1e3a8a", linewidth=1.5))

    # HLT merged / missing
    ax.add_patch(Circle((0.47, 0.58), 0.04, facecolor="#fca5a5", edgecolor="#7f1d1d", linewidth=2))
    ax.add_patch(Circle((0.55, 0.44), 0.03, facecolor="#fca5a5", edgecolor="#7f1d1d", linewidth=2))
    ax.text(0.43, 0.34, "drop / merge / smear", fontsize=12, color="#7f1d1d")

    # hard answer chosen
    for cx, cy, col in [(0.76, 0.62, "#86efac"), (0.82, 0.58, "#86efac"), (0.90, 0.56, "#86efac"), (0.78, 0.42, "#fde68a"), (0.87, 0.40, "#fde68a")]:
        ax.add_patch(Circle((cx, cy), 0.024, facecolor=col, edgecolor="#111827", linewidth=1.2))
    ax.add_patch(Rectangle((0.74, 0.36), 0.20, 0.33, fill=False, linestyle="--", linewidth=2.5, edgecolor="#b45309"))
    ax.text(0.71, 0.31, "hard output: pick ONE mapping", fontsize=12, color="#b45309")

    arr(ax, (0.31, 0.53), (0.365, 0.53), c="#1f2937")
    arr(ax, (0.635, 0.53), (0.69, 0.53), c="#1f2937")
    ax.text(0.39, 0.12, "Task: HLT -> OFFLINE reconstruction", fontsize=15, fontweight="bold", color="#0f172a")
    return fig


def slide2():
    fig, ax = setup("Soft View: What It Looks Like")

    panel(ax, 0.06, 0.16, 0.88, 0.70, "#ecfdf5", ec="#14532d")

    # Left: candidate pool
    panel(ax, 0.10, 0.30, 0.30, 0.46, "#dcfce7", ec="#166534")
    ax.text(0.13, 0.72, "Jet Candidate Pool", fontsize=17, fontweight="bold", color="#166534")
    ax.text(0.13, 0.68, "(cand_tokens + cand_weights)", fontsize=11, color="#166534")
    y0 = 0.62
    for i, w in enumerate([0.92, 0.78, 0.63, 0.47, 0.31, 0.18]):
        yy = y0 - i * 0.055
        ax.add_patch(Circle((0.14, yy), 0.010, facecolor="#22c55e", edgecolor="#14532d", linewidth=1.0))
        ax.add_patch(Rectangle((0.16, yy - 0.008), 0.19 * w, 0.016, facecolor="#22c55e", edgecolor="#14532d", linewidth=0.8))
    ax.text(0.12, 0.305, "one weighted superposition per jet", fontsize=11.5, color="#14532d")

    # Middle: projection
    panel(ax, 0.43, 0.42, 0.16, 0.20, "#bbf7d0", ec="#166534")
    ax.text(0.455, 0.54, "Project to", fontsize=12, color="#166534")
    ax.text(0.445, 0.50, "fixed L slots", fontsize=12, color="#166534")
    arr(ax, (0.40, 0.53), (0.43, 0.53), c="#166534", lw=3.0, ms=16)

    # Right: soft view tensor look
    panel(ax, 0.62, 0.24, 0.28, 0.54, "#d1fae5", ec="#166534")
    ax.text(0.65, 0.73, "Soft Corrected View", fontsize=17, fontweight="bold", color="#166534")
    ax.text(0.65, 0.69, "shape: [L, d]  (d=10 or 12)", fontsize=11, color="#166534")
    for r in range(7):
        for c in range(8):
            v = (r * 2 + c) % 4
            col = ["#bbf7d0", "#86efac", "#4ade80", "#22c55e"][v]
            ax.add_patch(Rectangle((0.65 + c * 0.026, 0.60 - r * 0.045), 0.020, 0.032, facecolor=col, edgecolor="#14532d", linewidth=0.4))
    ax.text(0.65, 0.26, "per-slot features include:\nkinematics + token weight +\nparent-added + eff-share", fontsize=11, color="#14532d")

    ax.text(0.09, 0.12, "Not multiple complete hypotheses. It is one jet-level weighted candidate set.", fontsize=14, color="#14532d", fontweight="bold")
    return fig


def slide3():
    fig, ax = setup("Hard View vs Soft View in Training")

    panel(ax, 0.06, 0.20, 0.41, 0.66, "#fef2f2", ec="#991b1b")
    panel(ax, 0.53, 0.20, 0.41, 0.66, "#ecfdf5", ec="#166534")

    ax.text(0.105, 0.80, "HARD VIEW", fontsize=20, fontweight="bold", color="#991b1b")
    ax.text(0.585, 0.80, "SOFT VIEW", fontsize=20, fontweight="bold", color="#166534")

    # hard path
    panel(ax, 0.10, 0.58, 0.12, 0.12, "#dbeafe", ec="#1e3a8a")
    panel(ax, 0.25, 0.58, 0.12, 0.12, "#ffedd5", ec="#9a3412")
    panel(ax, 0.13, 0.37, 0.22, 0.11, "#fecaca", ec="#991b1b")
    ax.text(0.16, 0.63, "HLT", fontsize=11, fontweight="bold", color="#1e3a8a")
    ax.text(0.266, 0.63, "hard\npick", fontsize=10, color="#9a3412", ha="center", va="center")
    ax.text(0.16, 0.41, "discrete bottleneck", fontsize=11, color="#991b1b")
    arr(ax, (0.22, 0.64), (0.25, 0.64), c="#991b1b", lw=2.4, ms=12)
    ax.text(0.10, 0.28, "commits early to one choice", fontsize=12, color="#7f1d1d")

    # soft path
    panel(ax, 0.57, 0.58, 0.12, 0.12, "#dbeafe", ec="#1e3a8a")
    panel(ax, 0.72, 0.58, 0.16, 0.12, "#dcfce7", ec="#166534")
    panel(ax, 0.62, 0.37, 0.22, 0.11, "#bbf7d0", ec="#166534")
    ax.text(0.62, 0.63, "HLT", fontsize=11, fontweight="bold", color="#1e3a8a")
    ax.text(0.80, 0.63, "soft\nweights", fontsize=10, color="#166534", ha="center", va="center")
    ax.text(0.655, 0.41, "differentiable path", fontsize=11, color="#166534")
    arr(ax, (0.69, 0.64), (0.72, 0.64), c="#166534", lw=2.4, ms=12)
    ax.text(0.57, 0.28, "keeps uncertainty during learning", fontsize=12, color="#14532d")

    # center divider and summary
    ax.plot([0.50, 0.50], [0.22, 0.84], color="#94a3b8", linewidth=2.0, linestyle="--")
    ax.text(0.17, 0.12, "single discrete correction", fontsize=13, fontweight="bold", color="#991b1b")
    ax.text(0.62, 0.12, "weighted candidate superposition", fontsize=13, fontweight="bold", color="#166534")
    return fig


def save(fig, p_png: Path, p_svg: Path):
    p_png.parent.mkdir(parents=True, exist_ok=True)
    p_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p_png, dpi=240, bbox_inches="tight")
    fig.savefig(p_svg, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("images"))
    args = ap.parse_args()
    d = args.out_dir

    save(slide1(), d / "soft_vs_hard_bold_slide1.png", d / "soft_vs_hard_bold_slide1.svg")
    save(slide2(), d / "soft_vs_hard_bold_slide2.png", d / "soft_vs_hard_bold_slide2.svg")
    save(slide3(), d / "soft_vs_hard_bold_slide3.png", d / "soft_vs_hard_bold_slide3.svg")
    print("Wrote bold slides to", d)


if __name__ == "__main__":
    main()
