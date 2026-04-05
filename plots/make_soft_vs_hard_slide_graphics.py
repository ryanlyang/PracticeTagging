#!/usr/bin/env python3
"""Create 3 conceptual graphics for soft-view vs hard-view slides."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, title, body, fc="#ffffff", ec="#1f2937", tsize=13, bsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.028, title, fontsize=tsize, fontweight="bold", va="top", color="#0f172a")
    ax.text(x + 0.012, y + h - 0.072, body, fontsize=bsize, va="top", linespacing=1.3, color="#1f2937")


def arrow(ax, s, e, color="#334155", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            s, e, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color, connectionstyle="arc3,rad=0.0"
        )
    )


def setup_figure(title: str):
    fig = plt.figure(figsize=(13.8, 7.8), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.text(0.03, 0.95, title, fontsize=21, fontweight="bold", color="#0b1220", va="top")
    return fig, ax


def slide1_hard_problem():
    fig, ax = setup_figure("Slide 1: Why Hard-View Mapping Fails")
    ax.text(
        0.03,
        0.905,
        "HLT applies irreversible transformations, so a single exact inverse mapping is often not identifiable.",
        fontsize=11,
        color="#334155",
        va="top",
    )

    box(
        ax,
        0.04,
        0.58,
        0.24,
        0.24,
        "Offline Truth (rich)",
        "Many constituents\nwith full detail.",
        fc="#e0f2fe",
    )
    box(
        ax,
        0.38,
        0.58,
        0.24,
        0.24,
        "HLT View (lossy)",
        "Dropped + merged + smeared.\nInformation is gone.",
        fc="#fee2e2",
    )
    box(
        ax,
        0.72,
        0.58,
        0.24,
        0.24,
        "Hard Inverse Target",
        "Forces one discrete answer\nfor an ambiguous inverse.",
        fc="#ffedd5",
    )

    arrow(ax, (0.28, 0.70), (0.38, 0.70))
    ax.text(0.305, 0.73, "HLT effects", fontsize=9.5, color="#475569")
    arrow(ax, (0.62, 0.70), (0.72, 0.70))
    ax.text(0.63, 0.73, "forced mapping", fontsize=9.5, color="#475569")

    # Ambiguity visual
    ax.text(0.04, 0.47, "Ambiguous inverse example:", fontsize=12.5, fontweight="bold", color="#0f172a")
    c1 = Circle((0.13, 0.31), 0.028, facecolor="#38bdf8", edgecolor="#0c4a6e", linewidth=1.0)
    c2 = Circle((0.19, 0.31), 0.028, facecolor="#38bdf8", edgecolor="#0c4a6e", linewidth=1.0)
    c3 = Circle((0.25, 0.31), 0.028, facecolor="#38bdf8", edgecolor="#0c4a6e", linewidth=1.0)
    ax.add_patch(c1); ax.add_patch(c2); ax.add_patch(c3)
    ax.text(0.095, 0.26, "Offline", fontsize=10, color="#0f172a")

    merged = Circle((0.48, 0.31), 0.04, facecolor="#fca5a5", edgecolor="#7f1d1d", linewidth=1.0)
    ax.add_patch(merged)
    ax.text(0.452, 0.26, "HLT", fontsize=10, color="#0f172a")
    arrow(ax, (0.29, 0.31), (0.43, 0.31))

    ax.text(0.58, 0.35, "Multiple plausible decompositions", fontsize=10.5, color="#0f172a")
    c4 = Circle((0.78, 0.36), 0.023, facecolor="#86efac", edgecolor="#14532d")
    c5 = Circle((0.83, 0.33), 0.023, facecolor="#86efac", edgecolor="#14532d")
    c6 = Circle((0.88, 0.29), 0.023, facecolor="#86efac", edgecolor="#14532d")
    c7 = Circle((0.78, 0.25), 0.023, facecolor="#fcd34d", edgecolor="#78350f")
    c8 = Circle((0.84, 0.25), 0.023, facecolor="#fcd34d", edgecolor="#78350f")
    ax.add_patch(c4); ax.add_patch(c5); ax.add_patch(c6); ax.add_patch(c7); ax.add_patch(c8)
    ax.text(0.66, 0.18, "Hard-view supervision picks one and treats it as certain.", fontsize=12, color="#7f1d1d")

    return fig


def slide2_soft_representation():
    fig, ax = setup_figure("Slide 2: Soft View Represents Ambiguity")
    ax.text(
        0.03,
        0.905,
        "Reconstructor predicts weighted candidate constituents; soft corrected view keeps uncertainty instead of collapsing it.",
        fontsize=11,
        color="#334155",
        va="top",
    )

    box(
        ax,
        0.05,
        0.56,
        0.27,
        0.29,
        "Reconstructor Output",
        "Candidate tokens + continuous weights\n(cand_tokens, cand_weights)\nNo hard keep/drop at this stage.",
        fc="#e0f2fe",
    )
    box(
        ax,
        0.37,
        0.56,
        0.27,
        0.29,
        "Soft Corrected View",
        "Fixed-length differentiable view:\n7 kinematic features + [tok_weight,\nparent_added_weight, eff_share].",
        fc="#dcfce7",
    )
    box(
        ax,
        0.69,
        0.56,
        0.26,
        0.29,
        "Dual-View Input",
        "Model consumes:\n(HLT view, soft corrected view)\nfor classification.",
        fc="#ede9fe",
    )
    arrow(ax, (0.32, 0.70), (0.37, 0.70))
    arrow(ax, (0.64, 0.70), (0.69, 0.70))

    # soft vs hard mini bars
    ax.text(0.06, 0.45, "Same ambiguous jet, two representations:", fontsize=12.5, fontweight="bold", color="#0f172a")
    box(
        ax,
        0.06,
        0.13,
        0.40,
        0.27,
        "Hard View",
        "Binary selection:\n[1, 0, 1, 0, ...]\nDrops alternate hypotheses.",
        fc="#fee2e2",
    )
    box(
        ax,
        0.52,
        0.13,
        0.42,
        0.27,
        "Soft View",
        "Continuous weighting:\n[0.82, 0.41, 0.77, 0.19, ...]\nRetains graded plausibility across candidates.",
        fc="#dcfce7",
    )

    ax.text(0.06, 0.09, "Key benefit: differentiable uncertainty representation during training.", fontsize=12, color="#14532d")
    return fig


def slide3_training_losses():
    fig, ax = setup_figure("Slide 3: How Soft-View Training and Losses Work")
    ax.text(
        0.03,
        0.905,
        "Classification gradients pass through the soft view into the reconstructor, alongside direct reconstruction constraints.",
        fontsize=11,
        color="#334155",
        va="top",
    )

    box(ax, 0.04, 0.57, 0.20, 0.24, "HLT View", "Input A\n(features, mask)", fc="#e0f2fe")
    box(ax, 0.28, 0.57, 0.20, 0.24, "Reconstructor", "Produces candidate\nweights/tokens", fc="#fef3c7")
    box(ax, 0.52, 0.57, 0.20, 0.24, "Soft View Builder", "Differentiable mapping\nto corrected view", fc="#dcfce7")
    box(ax, 0.76, 0.57, 0.20, 0.24, "Dual-View Head", "Classifier logits", fc="#ede9fe")
    arrow(ax, (0.24, 0.69), (0.28, 0.69))
    arrow(ax, (0.48, 0.69), (0.52, 0.69))
    arrow(ax, (0.72, 0.69), (0.76, 0.69))

    box(
        ax,
        0.70,
        0.22,
        0.26,
        0.20,
        "Classification Path",
        "L_cls (BCE)\n+ optional delta/rank terms\nfrom dual-view outputs.",
        fc="#fee2e2",
    )
    box(
        ax,
        0.36,
        0.22,
        0.30,
        0.20,
        "Reconstruction Path",
        "L_reco: set/phys/ratio/budget/local\n(+ optional discrepancy weighting).",
        fc="#ffedd5",
    )
    box(
        ax,
        0.04,
        0.22,
        0.28,
        0.20,
        "Coupling",
        "L_cons on reconstructor activity\n(+ anchor terms in prog. unfreeze).",
        fc="#e2e8f0",
    )

    arrow(ax, (0.86, 0.57), (0.83, 0.42), color="#7f1d1d")
    arrow(ax, (0.38, 0.57), (0.50, 0.42), color="#92400e")
    arrow(ax, (0.40, 0.57), (0.18, 0.42), color="#334155")

    ax.text(
        0.04,
        0.11,
        r"$L_{\mathrm{total}} = L_{\mathrm{cls}} + \lambda_{\mathrm{reco}}L_{\mathrm{reco}} + \lambda_{\mathrm{cons}}L_{\mathrm{cons}} + \lambda_{\delta}L_{\delta}\;(+\;\mathrm{optional\ terms})$",
        fontsize=17,
        color="#0b1220",
    )
    ax.text(
        0.04,
        0.07,
        "Result: reconstructor learns corrections that are both physically plausible and useful for downstream tagging.",
        fontsize=11.5,
        color="#14532d",
    )
    return fig


def save(fig, out_png: Path, out_svg: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate 3 soft-vs-hard conceptual slide graphics.")
    parser.add_argument("--out-dir", type=Path, default=Path("images"))
    args = parser.parse_args()

    out = args.out_dir
    save(
        slide1_hard_problem(),
        out / "soft_vs_hard_slide1_why_hard_fails.png",
        out / "soft_vs_hard_slide1_why_hard_fails.svg",
    )
    save(
        slide2_soft_representation(),
        out / "soft_vs_hard_slide2_soft_representation.png",
        out / "soft_vs_hard_slide2_soft_representation.svg",
    )
    save(
        slide3_training_losses(),
        out / "soft_vs_hard_slide3_training_losses.png",
        out / "soft_vs_hard_slide3_training_losses.svg",
    )

    print("Wrote slide graphics:")
    print(out / "soft_vs_hard_slide1_why_hard_fails.png")
    print(out / "soft_vs_hard_slide2_soft_representation.png")
    print(out / "soft_vs_hard_slide3_training_losses.png")


if __name__ == "__main__":
    main()

