# Three-Slide Deck: Reconstructor Loss + Meta-Fuser

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
