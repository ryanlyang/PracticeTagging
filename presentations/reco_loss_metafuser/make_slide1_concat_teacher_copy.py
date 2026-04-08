#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/presentations/reco_loss_metafuser"


def box(ax, x, y, w, h, txt, fc="#f5f7fb", ec="#2b2f3a", fs=12, weight="bold", lw=1.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight=weight, color="#111")


def arr(ax, x1, y1, x2, y2, color="#2b2f3a"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.8, color=color)
    ax.add_patch(a)


def base_canvas(title):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.04, 0.93, title, fontsize=28, weight="bold", color="#0f172a")
    return fig, ax


def make_assets():
    # Graphic only
    figg, axg = plt.subplots(figsize=(12, 4.5), dpi=180)
    axg.set_axis_off()
    axg.set_xlim(0, 1)
    axg.set_ylim(0, 1)

    box(axg, 0.03, 0.35, 0.16, 0.28, "HLT\nConstituents", fc="#dbeafe")
    box(axg, 0.25, 0.35, 0.2, 0.28, "Stage-A\nReconstructor", fc="#e0f2fe")
    box(axg, 0.51, 0.35, 0.2, 0.28, "Corrected\nView", fc="#ecfeff")
    box(
        axg,
        0.77,
        0.35,
        0.2,
        0.28,
        "Concatenated\nteacher",
        fc="#fff7cc",
        ec="#b45309",
        lw=2.8,
    )

    arr(axg, 0.19, 0.49, 0.25, 0.49)
    arr(axg, 0.45, 0.49, 0.51, 0.49)
    arr(axg, 0.71, 0.49, 0.77, 0.49)

    axg.text(
        0.5,
        0.86,
        "Same Stage-A pipeline, but teacher target switched to concatenated-teacher outputs",
        ha="center",
        fontsize=13,
        color="#1f2937",
    )

    figg.tight_layout()
    figg.savefig(f"{OUT_DIR}/graphic_1_intro_pipeline_concat_teacher.png", dpi=220)
    plt.close(figg)

    # Full slide
    fig, ax = base_canvas("Slide 1: Offline Teacher Loss -> Concatenated Teacher Loss")
    ax.text(
        0.04,
        0.84,
        "Goal: keep Stage-A reconstructor, but replace offline-teacher target with concatenated-teacher target.",
        fontsize=16,
        color="#111827",
    )

    bullets = [
        "Inputs remain HLT constituents; reconstructor still produces a corrected view",
        "Change is only the supervisory teacher target: concatenated teacher instead of offline teacher",
        "Loss terms still compare teacher outputs (logits/embeddings/tokens), now against concat-teacher reference",
    ]
    y = 0.77
    for b in bullets:
        ax.text(0.06, y, f"- {b}", fontsize=14, color="#111827")
        y -= 0.055

    gax = fig.add_axes([0.06, 0.18, 0.88, 0.45])
    gax.set_axis_off()
    gax.set_xlim(0, 1)
    gax.set_ylim(0, 1)

    box(gax, 0.03, 0.35, 0.16, 0.28, "HLT\nConstituents", fc="#dbeafe")
    box(gax, 0.25, 0.35, 0.2, 0.28, "Stage-A\nReconstructor", fc="#e0f2fe")
    box(gax, 0.51, 0.35, 0.2, 0.28, "Corrected\nView", fc="#ecfeff")
    box(
        gax,
        0.77,
        0.35,
        0.2,
        0.28,
        "Concatenated\nteacher",
        fc="#fff7cc",
        ec="#b45309",
        lw=2.8,
    )

    arr(gax, 0.19, 0.49, 0.25, 0.49)
    arr(gax, 0.45, 0.49, 0.51, 0.49)
    arr(gax, 0.71, 0.49, 0.77, 0.49)

    gax.text(
        0.5,
        0.12,
        "Only target swap: offline-teacher supervision -> concatenated-teacher supervision",
        ha="center",
        fontsize=12,
        color="#334155",
    )

    fig.savefig(f"{OUT_DIR}/slide_1_intro_concat_teacher.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print(f"{OUT_DIR}/graphic_1_intro_pipeline_concat_teacher.png")
    print(f"{OUT_DIR}/slide_1_intro_concat_teacher.png")


if __name__ == "__main__":
    make_assets()
