#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/presentations/reco_loss_metafuser"


def box(ax, x, y, w, h, txt, fc="#f5f7fb", ec="#2b2f3a", fs=12, weight="bold"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=1.6)
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


def slide1_and_graphic():
    # Graphic
    figg, axg = plt.subplots(figsize=(12, 4.5), dpi=180)
    axg.set_axis_off()
    axg.set_xlim(0, 1)
    axg.set_ylim(0, 1)
    box(axg, 0.03, 0.35, 0.16, 0.28, "HLT\nConstituents", fc="#dbeafe")
    box(axg, 0.25, 0.35, 0.2, 0.28, "Stage-A\nReconstructor", fc="#e0f2fe")
    box(axg, 0.51, 0.35, 0.2, 0.28, "Corrected\nView", fc="#ecfeff")
    box(axg, 0.77, 0.35, 0.2, 0.28, "Teacher\nScoring", fc="#f1f5f9")
    arr(axg, 0.19, 0.49, 0.25, 0.49)
    arr(axg, 0.45, 0.49, 0.51, 0.49)
    arr(axg, 0.71, 0.49, 0.77, 0.49)
    axg.text(0.5, 0.86, "Model-4 path: HLT -> reconstructed corrected view -> teacher-aligned score", ha="center", fontsize=13, color="#1f2937")
    figg.tight_layout()
    figg.savefig(f"{OUT_DIR}/graphic_1_intro_pipeline.png", dpi=220)
    plt.close(figg)

    # Slide
    fig, ax = base_canvas("Slide 1: What This Setup Is Doing")
    ax.text(0.04, 0.84, "Goal: recover offline-like tagging performance from HLT inputs using a Stage-A reconstructor.", fontsize=16, color="#111827")
    bullets = [
        "Run script: sbatch/.../run_m4_recoteacher_s01_corrected_150k75k150k.sh",
        "Trains teacher + HLT baseline, then trains reconstructor with teacher-guided losses",
        "After Stage-A, builds corrected view and evaluates/improves downstream tagging",
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
    box(gax, 0.77, 0.35, 0.2, 0.28, "Teacher\nScoring", fc="#f1f5f9")
    arr(gax, 0.19, 0.49, 0.25, 0.49)
    arr(gax, 0.45, 0.49, 0.51, 0.49)
    arr(gax, 0.71, 0.49, 0.77, 0.49)
    gax.text(0.5, 0.12, "Stage-scale curriculum during Stage-A: 0.35 -> 0.70 -> 1.00", ha="center", fontsize=12, color="#334155")

    fig.savefig(f"{OUT_DIR}/slide_1_intro.png", dpi=220)
    plt.close(fig)


def slide2_and_graphic():
    # Graphic (loss block diagram)
    figg, axg = plt.subplots(figsize=(12, 5), dpi=180)
    axg.set_axis_off()
    axg.set_xlim(0, 1)
    axg.set_ylim(0, 1)
    box(axg, 0.05, 0.58, 0.2, 0.28, "Teacher-guided\nKD + Emb + Tok", fc="#fee2e2")
    box(axg, 0.30, 0.58, 0.2, 0.28, "Physics\nReconstruction", fc="#ffedd5")
    box(axg, 0.55, 0.58, 0.2, 0.28, "Budget Hinge\n(edit budget)", fc="#fef3c7")
    box(axg, 0.80, 0.58, 0.15, 0.28, "L_delta\n(optional)", fc="#dcfce7")
    box(axg, 0.35, 0.12, 0.3, 0.28, "Total Stage-A Loss", fc="#e2e8f0", fs=13)
    arr(axg, 0.15, 0.58, 0.42, 0.40)
    arr(axg, 0.40, 0.58, 0.50, 0.40)
    arr(axg, 0.65, 0.58, 0.58, 0.40)
    arr(axg, 0.875, 0.58, 0.62, 0.40)
    axg.text(0.5, 0.93, "L_total = weighted sum of normalized terms (+ optional complement term)", ha="center", fontsize=13, color="#1f2937")
    figg.tight_layout()
    figg.savefig(f"{OUT_DIR}/graphic_2_exact_losses.png", dpi=220)
    plt.close(figg)

    # Slide
    fig, ax = base_canvas("Slide 2: Exact Stage-A Loss (Model-4 Run)")
    eq_lines = [
        r"$L_{\mathrm{total}} = \lambda_{kd} \tilde{L}_{kd} + \lambda_{emb} \tilde{L}_{emb} + \lambda_{tok} \tilde{L}_{tok} + \lambda_{phys} \tilde{L}_{phys} + \lambda_{budget} \tilde{L}_{budget\_hinge} + \lambda_{\Delta} L_{\Delta}$",
        r"with EMA-normalized terms $\tilde{L}_i$ (decay=0.98, eps=1e-6).",
    ]
    ax.text(0.04, 0.84, eq_lines[0], fontsize=15, color="#111827")
    ax.text(0.04, 0.79, eq_lines[1], fontsize=13, color="#334155")

    bullets = [
        "Run-time weights from run_m4: lambda_kd=1.0, lambda_emb=1.2, lambda_tok=0.6, lambda_phys=0.2, lambda_budget_hinge=0.03",
        "Budget controls: budget_eps=0.015, budget_weight_floor=1e-4, kd_temperature=2.5",
        "Delta term form: L_delta = -gain + delta_lambda_fp * cost, but here lambda_delta=0.00 (disabled in this run)",
    ]
    y = 0.72
    for b in bullets:
        ax.text(0.06, y, f"- {b}", fontsize=13.5, color="#111827")
        y -= 0.06

    gax = fig.add_axes([0.06, 0.12, 0.88, 0.38])
    gax.set_axis_off()
    gax.set_xlim(0, 1)
    gax.set_ylim(0, 1)
    box(gax, 0.05, 0.55, 0.2, 0.3, "KD + Emb + Tok", fc="#fee2e2")
    box(gax, 0.30, 0.55, 0.2, 0.3, "Physics", fc="#ffedd5")
    box(gax, 0.55, 0.55, 0.2, 0.3, "Budget Hinge", fc="#fef3c7")
    box(gax, 0.80, 0.55, 0.15, 0.3, "Delta\n(optional)", fc="#dcfce7")
    box(gax, 0.35, 0.10, 0.3, 0.3, "Stage-A\nTotal", fc="#e2e8f0")
    arr(gax, 0.15, 0.55, 0.43, 0.40)
    arr(gax, 0.40, 0.55, 0.50, 0.40)
    arr(gax, 0.65, 0.55, 0.58, 0.40)
    arr(gax, 0.875, 0.55, 0.62, 0.40)

    fig.savefig(f"{OUT_DIR}/slide_2_exact_losses.png", dpi=220)
    plt.close(fig)


def slide3_and_graphic():
    # Graphic (meta-fuser)
    figg, axg = plt.subplots(figsize=(12, 5), dpi=180)
    axg.set_axis_off()
    axg.set_xlim(0, 1)
    axg.set_ylim(0, 1)
    box(axg, 0.03, 0.58, 0.14, 0.25, "HLT score", fc="#dbeafe")
    box(axg, 0.20, 0.58, 0.14, 0.25, "Joint score", fc="#e0e7ff")
    box(axg, 0.37, 0.58, 0.14, 0.25, "RecoTeacher\nscore", fc="#fce7f3")
    box(axg, 0.54, 0.58, 0.20, 0.25, "|HLT-Joint|, |HLT-Reco|", fc="#f3e8ff")
    box(axg, 0.78, 0.58, 0.18, 0.25, "LogReg\nMeta-Fuser", fc="#dcfce7")
    arr(axg, 0.17, 0.70, 0.78, 0.70)
    arr(axg, 0.34, 0.66, 0.78, 0.66)
    arr(axg, 0.51, 0.62, 0.78, 0.62)
    arr(axg, 0.74, 0.58, 0.78, 0.58)
    box(axg, 0.36, 0.14, 0.28, 0.26, "Threshold @ target TPR\n(from val)\n-> lower test FPR", fc="#e2e8f0")
    arr(axg, 0.87, 0.58, 0.50, 0.40)
    axg.text(0.5, 0.92, "Meta-fuser combines complementary errors from HLT + Joint + RecoTeacher", ha="center", fontsize=13, color="#1f2937")
    figg.tight_layout()
    figg.savefig(f"{OUT_DIR}/graphic_3_meta_fuser.png", dpi=220)
    plt.close(figg)

    # Slide
    fig, ax = base_canvas("Slide 3: Meta-Fuser to Improve Over HLT + Joint")
    ax.text(0.04, 0.84, "Used in fusion analysis to reduce FPR at fixed TPR (typically TPR=0.50).", fontsize=16, color="#111827")

    bullets = [
        "Features (from analyze_hlt_joint_recoteacher_fusion.py): [HLT, Joint, RecoTeacher, |HLT-Joint|, |HLT-RecoTeacher|]",
        "Model: class-balanced Logistic Regression; C selected on a val-split by minimizing FPR at target TPR",
        "Then refit on full val, set threshold from val target TPR, and evaluate on test",
        "Why it helps: exploits disagreement structure, not just averaging scores",
    ]
    y = 0.77
    for b in bullets:
        ax.text(0.06, y, f"- {b}", fontsize=13.5, color="#111827")
        y -= 0.06

    gax = fig.add_axes([0.05, 0.11, 0.90, 0.42])
    gax.set_axis_off()
    gax.set_xlim(0, 1)
    gax.set_ylim(0, 1)
    box(gax, 0.03, 0.56, 0.14, 0.28, "HLT", fc="#dbeafe")
    box(gax, 0.20, 0.56, 0.14, 0.28, "Joint", fc="#e0e7ff")
    box(gax, 0.37, 0.56, 0.14, 0.28, "RecoTeacher", fc="#fce7f3")
    box(gax, 0.54, 0.56, 0.20, 0.28, "|HLT-J|, |HLT-R|", fc="#f3e8ff")
    box(gax, 0.78, 0.56, 0.18, 0.28, "Meta-Fuser\n(LogReg)", fc="#dcfce7")
    arr(gax, 0.17, 0.70, 0.78, 0.70)
    arr(gax, 0.34, 0.66, 0.78, 0.66)
    arr(gax, 0.51, 0.62, 0.78, 0.62)
    arr(gax, 0.74, 0.58, 0.78, 0.58)
    box(gax, 0.36, 0.10, 0.28, 0.30, "Val-selected threshold\n@ target TPR\n=> test FPR drop", fc="#e2e8f0")
    arr(gax, 0.87, 0.56, 0.50, 0.40)

    fig.savefig(f"{OUT_DIR}/slide_3_meta_fuser.png", dpi=220)
    plt.close(fig)


def write_notes_md():
    txt = """# Three-Slide Deck: Reconstructor Loss + Meta-Fuser

## Slide 1 (Intro)
- File: `slide_1_intro.png`
- Graphic: `graphic_1_intro_pipeline.png`
- Message: Stage-A reconstructor learns to transform HLT-level view toward an offline-like corrected view for better tagging.

## Slide 2 (Exact Losses)
- File: `slide_2_exact_losses.png`
- Graphic: `graphic_2_exact_losses.png`
- Exact run parameters from:
  - `sbatch/reco_teacher_joint_fusion_6model_150k75k150k/run_m4_recoteacher_s01_corrected_150k75k150k.sh`
- Active Stage-A weights in this run:
  - `lambda_kd=1.0`, `lambda_emb=1.2`, `lambda_tok=0.6`, `lambda_phys=0.2`, `lambda_budget_hinge=0.03`
  - `lambda_delta=0.00` (delta term present in code but disabled in this run)

## Slide 3 (Meta-Fuser)
- File: `slide_3_meta_fuser.png`
- Graphic: `graphic_3_meta_fuser.png`
- Features from `analyze_hlt_joint_recoteacher_fusion.py`:
  - `[HLT, Joint, RecoTeacher, |HLT-Joint|, |HLT-RecoTeacher|]`
- Model: class-balanced logistic regression, val-split model selection by FPR at target TPR, then test eval.
"""
    with open(f"{OUT_DIR}/slides_notes.md", "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    slide1_and_graphic()
    slide2_and_graphic()
    slide3_and_graphic()
    write_notes_md()
    print("Wrote slide assets to", OUT_DIR)


if __name__ == "__main__":
    main()
