#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "graphic_1_intro_pipeline_concat_teacher_offline_style.png"

fig, ax = plt.subplots(figsize=(12.92, 6.68), dpi=100)
fig.patch.set_facecolor("#e6e6e8")
ax.set_facecolor("#e6e6e8")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, title, subtitle="", fc="#d7e3f3", ec="#1f2a44", lw=1.6, title_fs=12, sub_fs=8.5):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_fs, weight="bold", color="#111827")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.35, subtitle, ha="center", va="center", fontsize=sub_fs, color="#334155")


def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=2.0, color="#334155")
    ax.add_patch(a)

ax.text(0.5, 0.94, "Offline Teacher-Based Loss", ha="center", va="center", fontsize=14, weight="bold", color="#111827")

# left to middle pipeline
box(0.03, 0.57, 0.18, 0.28, "HLT Input View", "features + constituents", fc="#c3d3e6")
box(0.25, 0.57, 0.20, 0.28, "Reconstructor", "predict corrected view", fc="#c2d3e1")
box(0.25, 0.27, 0.20, 0.21, "Concatenated View", "ground-truth reference", fc="#ffd6d6", ec="#991b1b", lw=2.4)

# teacher comparison branch (highlighted)
box(0.48, 0.65, 0.20, 0.18, "Concatenated Teacher", "logit_r, emb_r", fc="#ffd6d6", ec="#991b1b", lw=2.6)
box(0.48, 0.40, 0.20, 0.18, "Concatenated Teacher", "logit_c, emb_c", fc="#ffd6d6", ec="#991b1b", lw=2.6)

box(0.72, 0.51, 0.17, 0.22, "Compare Teacher\nOutputs", "logit diff + embedding diff\n(+ attention/token diff)", fc="#ececec", ec="#5b6f92", title_fs=11.2, sub_fs=7.3)
box(0.78, 0.09, 0.20, 0.26, "Reconstructor\nLoss", "existing loss terms\n(set matching + budget + ... )\n+ Teacher_Comparison_terms", fc="#c9d2de", ec="#1f2a44", title_fs=11.2, sub_fs=7.4)

# arrows
arrow(0.21, 0.69, 0.25, 0.69)   # HLT -> reconstructor
arrow(0.44, 0.69, 0.48, 0.76)   # reconstructor -> teacher reco
arrow(0.45, 0.35, 0.49, 0.54)   # offline -> concatenated teacher
arrow(0.68, 0.73, 0.72, 0.64)   # teacher reco -> compare
arrow(0.68, 0.52, 0.72, 0.60)   # concat teacher -> compare
arrow(0.83, 0.51, 0.84, 0.34)   # compare -> loss

fig.tight_layout(pad=0.5)
fig.savefig(OUT, dpi=100)
print("wrote", OUT)
