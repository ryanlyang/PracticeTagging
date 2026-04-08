#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "graphic_concat_loss_calculation_focus.png"

fig, ax = plt.subplots(figsize=(13.2, 7.2), dpi=140)
fig.patch.set_facecolor("#e6e6e8")
ax.set_facecolor("#e6e6e8")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, title, subtitle="", fc="#dbe7f5", ec="#1f2a44", lw=1.8, tfs=13, sfs=9.2, sw="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h*0.64, title, ha="center", va="center", fontsize=tfs, weight="bold", color="#111827")
    if subtitle:
        ax.text(x + w/2, y + h*0.34, subtitle, ha="center", va="center", fontsize=sfs, color="#334155", weight=sw)


def arr(x1, y1, x2, y2, c="#334155", lw=2.1):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=14, lw=lw, color=c))


def arr_curve(x1, y1, x2, y2, c="#334155", lw=2.1, cs="angle3,angleA=0,angleB=-90"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=lw,
            color=c,
            connectionstyle=cs,
        )
    )

ax.text(0.5, 0.95, "How Loss Is Computed: Compare Concat(HLT+Reco) vs Concat(HLT+Offline)",
        ha="center", va="center", fontsize=18, weight="bold", color="#111827")

# Left two concat views
box(0.06, 0.63, 0.27, 0.20, "Predicted Concat View", "[HLT + Reco Additions]", fc="#ffd6d6", ec="#991b1b", lw=2.5)
box(0.06, 0.35, 0.27, 0.20, "Target Concat View", "[HLT + Offline]", fc="#ffe9c7", ec="#9a6a10", lw=2.3)

# Shared teacher
box(0.41, 0.50, 0.20, 0.22, "Same\nConcatenated\nTeacher", "shared weights\nfor both paths", fc="#f6f0ff", ec="#5b3b8a", lw=2.2, tfs=12.2, sfs=8.8)

# outputs
box(0.69, 0.64, 0.22, 0.15, "Teacher Outputs (Pred)", "logit_p, emb_p,\nattn_p", fc="#eef2ff", ec="#3f4f7a")
box(0.69, 0.39, 0.22, 0.15, "Teacher Outputs (Target)", "logit_t, emb_t,\nattn_t", fc="#eef2ff", ec="#3f4f7a")

# compare + loss
box(0.69, 0.15, 0.22, 0.17, "Comparison Terms", "KD(logit_p, logit_t)\nEmb(emb_p, emb_t)\nTok(attn_p, attn_t)", fc="#ececec", ec="#5b6f92", tfs=12, sfs=9.6, sw="bold")
box(0.38, 0.12, 0.24, 0.16, "Reconstructor Loss", "L = L_teacher_match\n+ L_phys + L_budget", fc="#c9d2de", ec="#1f2a44", tfs=12, sfs=9.2)

# arrows
arr(0.33, 0.73, 0.41, 0.63)
arr(0.33, 0.45, 0.41, 0.58)
arr(0.61, 0.63, 0.69, 0.72)
arr(0.61, 0.58, 0.69, 0.47)
# Route arrows separately so they do not pass through boxes.
arr(0.80, 0.39, 0.83, 0.32)
arr(0.69, 0.23, 0.62, 0.20)

# Continuous routed connector from pred-output to comparison right edge.
arr_curve(0.91, 0.715, 0.91, 0.235)

fig.tight_layout(pad=0.7)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
