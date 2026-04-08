#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "graphic_reconstructor_as_adder.png"

fig, ax = plt.subplots(figsize=(13.0, 7.0), dpi=140)
fig.patch.set_facecolor("#e6e6e8")
ax.set_facecolor("#e6e6e8")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, title, subtitle="", fc="#dbe7f5", ec="#1f2a44", lw=1.6, title_fs=12, sub_fs=9, weight="bold"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_fs, weight=weight, color="#111827")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.34, subtitle, ha="center", va="center", fontsize=sub_fs, color="#334155")


def arr(x1, y1, x2, y2, color="#334155", lw=2.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color)
    ax.add_patch(a)


ax.text(0.5, 0.95, "Reconstructor as an Adder (Not a Replacer)", ha="center", va="center", fontsize=19, weight="bold", color="#111827")

# Left/base path
box(0.04, 0.60, 0.18, 0.22, "HLT Base View", "kept as-is", fc="#c7d7eb")
box(0.29, 0.60, 0.21, 0.22, "Reconstructor", "predict residual additions", fc="#c1d7ea")
box(0.29, 0.30, 0.21, 0.18, "Reco Additions", "unsmear + unmerge + reassign", fc="#ffd6d6", ec="#991b1b", lw=2.5)

arr(0.22, 0.71, 0.29, 0.71)
arr(0.40, 0.60, 0.40, 0.48)

# Pred concat path
box(0.56, 0.60, 0.19, 0.22, "Concat Pred View", "[HLT + Reco Additions]", fc="#ffd6d6", ec="#991b1b", lw=2.5)
arr(0.50, 0.71, 0.56, 0.71)
arr(0.50, 0.39, 0.56, 0.64)

# Target concat path
box(0.56, 0.30, 0.19, 0.18, "Concat Target View", "[HLT + Offline]", fc="#efe5be", ec="#6b5d1f", lw=2.2)

# Teacher / compare / loss
box(0.79, 0.60, 0.17, 0.22, "Concatenated Teacher", "logits + embeddings + attention", fc="#ffd6d6", ec="#991b1b", lw=2.5)
box(0.79, 0.30, 0.17, 0.18, "Compare Outputs", "pred vs target teacher signals", fc="#ececec", ec="#5b6f92")
box(0.67, 0.07, 0.29, 0.16, "Reconstructor Loss", "teacher-match terms + physics + budget", fc="#c9d2de", ec="#1f2a44", title_fs=12, sub_fs=9)

arr(0.75, 0.71, 0.79, 0.71)
arr(0.75, 0.39, 0.79, 0.39)
arr(0.875, 0.60, 0.875, 0.48)
arr(0.875, 0.30, 0.83, 0.23)

# Emphasis note
ax.text(
    0.05,
    0.10,
    "Key idea: HLT tokens remain; reconstructor only adds missing information that closes\n"
    "the teacher gap between concat(HLT+reco_additions) and concat(HLT+offline).",
    ha="left",
    va="center",
    fontsize=10,
    color="#1f2937",
)

fig.tight_layout(pad=0.6)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
