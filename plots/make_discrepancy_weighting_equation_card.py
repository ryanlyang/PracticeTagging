#!/usr/bin/env python3
"""Generate a simple equation card for discrepancy weighting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def make_figure() -> plt.Figure:
    fig = plt.figure(figsize=(13.5, 7.6), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    # Main card
    card = FancyBboxPatch(
        (0.03, 0.05),
        0.94,
        0.90,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
    )
    ax.add_patch(card)

    ax.text(
        0.06,
        0.92,
        "Discrepancy Weighting (smooth_delta mode)",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
        va="top",
    )

    ax.text(
        0.06,
        0.84,
        r"$r_i = \max\!\left(p_i^{\mathrm{HLT}} - p_i^{\mathrm{teacher}},\,0\right)\,(1-y_i)\,g_i$",
        fontsize=28,
        color="#0b1220",
        va="top",
    )

    ax.text(
        0.06,
        0.66,
        r"$\tilde{w}_i = \mathrm{clip}\!\left(1 + \lambda\,r_i,\;1,\;m\right), \qquad"
        r"w_i = \frac{\tilde{w}_i}{\frac{1}{N}\sum_{j=1}^{N}\tilde{w}_j}$",
        fontsize=24,
        color="#0b1220",
        va="top",
    )

    ax.text(
        0.06,
        0.48,
        r"$\mathcal{L}_{\mathrm{weighted}} = \frac{\sum_i w_i\,\ell_i}{\sum_i w_i}$",
        fontsize=30,
        color="#0b1220",
        va="top",
    )

    expl = (
        "Where:\n"
        "y_i: true class (0 = background, 1 = signal)\n"
        "g_i: teacher gate (hard-correct × confidence × teacher-better)\n"
        "lambda: discrepancy strength, m: max weight cap\n"
        "ell_i: per-jet loss term (reco loss or BCE)\n\n"
        "Effect:\n"
        "Jets where HLT is more signal-like than teacher (especially negatives) get larger weight.\n"
        "Mean-normalization keeps average weight near 1, so it re-distributes emphasis rather than just scaling total loss."
    )
    ax.text(
        0.06,
        0.34,
        expl,
        fontsize=14,
        color="#1e293b",
        va="top",
        linespacing=1.35,
    )

    defaults = (
        "This runner defaults: reco (lambda=6, max=8), cls (lambda=2, max=3), "
        "target_tpr=0.50, tau=0.05, mean-normalize ON, include_pos OFF."
    )
    ax.text(
        0.06,
        0.11,
        defaults,
        fontsize=12.5,
        color="#334155",
        va="top",
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create discrepancy-weighting equation card.")
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("images/discrepancy_weighting_equation_card.png"),
        help="Output PNG file.",
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=Path("images/discrepancy_weighting_equation_card.svg"),
        help="Output SVG file.",
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
