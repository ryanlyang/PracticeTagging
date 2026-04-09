#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JetClass joint dual-view:
- m2-style HLT corruption (dominant-token identity on merges)
- jet-latent set2set reconstructor
- progressive-unfreeze + anchor-constrained Stage-C

This entrypoint patches behavior without modifying the base JetClass script.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_jetclass_joint_dualview_stage2_unmergeonly as base
import train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt as m2hlt
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_jetlatent_set2set as jl
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as reco_joint


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


STAGEC_PROGRESSIVE_UNFREEZE = _env_bool("JETCLASS_STAGEC_PROGRESSIVE_UNFREEZE", True)
STAGEC_UNFREEZE_PHASE1_EPOCHS = int(os.getenv("JETCLASS_STAGEC_UNFREEZE_PHASE1_EPOCHS", "3"))
STAGEC_UNFREEZE_PHASE2_EPOCHS = int(os.getenv("JETCLASS_STAGEC_UNFREEZE_PHASE2_EPOCHS", "7"))
STAGEC_UNFREEZE_LAST_N_ENCODER_LAYERS = int(os.getenv("JETCLASS_STAGEC_UNFREEZE_LAST_N_ENCODER_LAYERS", "2"))
STAGEC_LAMBDA_PARAM_ANCHOR = float(os.getenv("JETCLASS_STAGEC_LAMBDA_PARAM_ANCHOR", "0.02"))
STAGEC_LAMBDA_OUTPUT_ANCHOR = float(os.getenv("JETCLASS_STAGEC_LAMBDA_OUTPUT_ANCHOR", "0.02"))
STAGEC_ANCHOR_DECAY = float(os.getenv("JETCLASS_STAGEC_ANCHOR_DECAY", "0.97"))
STAGEC_RECO_RAMP_EPOCHS = int(os.getenv("JETCLASS_STAGEC_RECO_RAMP_EPOCHS", "8"))


def _set_reco_trainable_progressive(model: nn.Module, ep: int) -> None:
    # Freeze all, then progressively unfreeze.
    for p in model.parameters():
        p.requires_grad_(False)

    always_patterns = (
        "action_head",
        "unsmear_head",
        "reassign_head",
        "split_exist_head",
        "split_delta_head",
        "budget_head",
        "gen_attn",
        "gen_norm",
        "gen_head",
        "gen_exist_head",
        "pool_attn",
        "pool_query",
        "gen_queries",
        "token_norm",
    )
    for n, p in model.named_parameters():
        if any(k in n for k in always_patterns):
            p.requires_grad_(True)

    if ep >= int(STAGEC_UNFREEZE_PHASE1_EPOCHS):
        if hasattr(model, "input_proj"):
            for p in model.input_proj.parameters():
                p.requires_grad_(True)
        if hasattr(model, "relpos_mlp"):
            for p in model.relpos_mlp.parameters():
                p.requires_grad_(True)
        if hasattr(model, "encoder_layers"):
            layers = list(model.encoder_layers)
            k = max(0, int(STAGEC_UNFREEZE_LAST_N_ENCODER_LAYERS))
            start = max(0, len(layers) - k)
            for li in range(start, len(layers)):
                for p in layers[li].parameters():
                    p.requires_grad_(True)

    if ep >= int(STAGEC_UNFREEZE_PHASE2_EPOCHS):
        for p in model.parameters():
            p.requires_grad_(True)


def train_joint_dual_multiclass_with_stagec_controls(
    reconstructor: nn.Module,
    dual_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    min_epochs: int,
    lr_dual: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_cons: float,
    loss_cfg: Dict,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    corrected_weight_floor: float,
) -> Tuple[nn.Module, nn.Module, Dict[str, float], Dict[str, Dict[str, torch.Tensor]]]:
    stagec_mode = (not freeze_reconstructor) and str(stage_name).startswith("StageC")

    for p in dual_model.parameters():
        p.requires_grad_(True)
    if freeze_reconstructor:
        for p in reconstructor.parameters():
            p.requires_grad_(False)
    else:
        for p in reconstructor.parameters():
            p.requires_grad_(True)

    params = [{"params": dual_model.parameters(), "lr": float(lr_dual)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))

    def _lr_lambda(ep: int) -> float:
        if ep < int(warmup_epochs):
            return (ep + 1) / max(int(warmup_epochs), 1)
        x = (ep - int(warmup_epochs)) / max(int(epochs) - int(warmup_epochs), 1)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    # Optional anchors (captured at Stage-C start).
    lambda_param_anchor = float(max(STAGEC_LAMBDA_PARAM_ANCHOR, 0.0)) if stagec_mode else 0.0
    lambda_output_anchor = float(max(STAGEC_LAMBDA_OUTPUT_ANCHOR, 0.0)) if stagec_mode else 0.0
    anchor_decay = float(max(STAGEC_ANCHOR_DECAY, 0.0))
    anchor_reco_params: Dict[str, torch.Tensor] = {}
    anchor_dual_params: Dict[str, torch.Tensor] = {}
    ref_reco_model = None
    ref_dual_model = None

    if stagec_mode and (lambda_param_anchor > 0.0 or lambda_output_anchor > 0.0):
        anchor_reco_params = {n: p.detach().clone() for n, p in reconstructor.named_parameters()}
        anchor_dual_params = {n: p.detach().clone() for n, p in dual_model.named_parameters()}
        if lambda_output_anchor > 0.0:
            ref_reco_model = copy.deepcopy(reconstructor).to(device).eval()
            ref_dual_model = copy.deepcopy(dual_model).to(device).eval()
            for p in ref_reco_model.parameters():
                p.requires_grad_(False)
            for p in ref_dual_model.parameters():
                p.requires_grad_(False)

    best_metric = float("-inf")
    best_val_auc = float("-inf")
    best_val_acc = float("-inf")
    best_state_dual = None
    best_state_reco = None
    wait = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()
        dual_model.train()

        if stagec_mode and bool(STAGEC_PROGRESSIVE_UNFREEZE):
            _set_reco_trainable_progressive(reconstructor, ep)

        tr_total = tr_cls = tr_reco = tr_cons = tr_anchor = 0.0
        n_tr = 0
        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            feat_hlt_dual = batch["feat_hlt_dual"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_b, mask_b = base.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b)
            loss_cls = F.cross_entropy(logits, y)

            if float(lambda_reco) > 0.0:
                losses_reco = base.compute_reconstruction_losses_weighted(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    loss_cfg,
                    sample_weight=None,
                )
                loss_reco = losses_reco["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            # Stage-C reco ramp.
            if stagec_mode and int(STAGEC_RECO_RAMP_EPOCHS) > 0:
                reco_scale = min(1.0, float(ep + 1) / float(max(int(STAGEC_RECO_RAMP_EPOCHS), 1)))
            else:
                reco_scale = 1.0
            lambda_reco_eff = float(lambda_reco) * float(reco_scale)

            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            loss_anchor_param = torch.zeros((), device=device)
            if stagec_mode and lambda_param_anchor > 0.0:
                acc = torch.zeros((), device=device)
                cnt = 0
                for n, p in reconstructor.named_parameters():
                    if n in anchor_reco_params:
                        acc = acc + F.mse_loss(p, anchor_reco_params[n], reduction="mean")
                        cnt += 1
                for n, p in dual_model.named_parameters():
                    if n in anchor_dual_params:
                        acc = acc + F.mse_loss(p, anchor_dual_params[n], reduction="mean")
                        cnt += 1
                if cnt > 0:
                    loss_anchor_param = acc / float(cnt)

            loss_anchor_out = torch.zeros((), device=device)
            if stagec_mode and lambda_output_anchor > 0.0 and ref_reco_model is not None and ref_dual_model is not None:
                with torch.no_grad():
                    ref_reco_out = ref_reco_model(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
                    ref_feat_b, ref_mask_b = base.build_soft_corrected_view(
                        ref_reco_out,
                        weight_floor=float(corrected_weight_floor),
                        scale_features_by_weight=True,
                        include_flags=False,
                    )
                    ref_logits = ref_dual_model(feat_hlt_dual, mask_hlt, ref_feat_b, ref_mask_b)
                loss_anchor_out = F.smooth_l1_loss(logits, ref_logits, reduction="mean")

            anchor_scale = (float(anchor_decay) ** float(ep)) if stagec_mode else 1.0
            loss = (
                loss_cls
                + float(lambda_reco_eff) * loss_reco
                + float(lambda_cons) * loss_cons
                + float(anchor_scale) * float(lambda_param_anchor) * loss_anchor_param
                + float(anchor_scale) * float(lambda_output_anchor) * loss_anchor_out
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = int(y.shape[0])
            tr_total += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_reco += float(loss_reco.item()) * bs
            tr_cons += float(loss_cons.item()) * bs
            tr_anchor += float((loss_anchor_param + loss_anchor_out).item()) * bs
            n_tr += bs

        sch.step()

        tr_total /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)
        tr_anchor /= max(n_tr, 1)

        va = base.eval_joint_multiclass(
            reconstructor=reconstructor,
            dual_model=dual_model,
            loader=val_loader,
            device=device,
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
            corrected_weight_floor=float(corrected_weight_floor),
        )

        va_auc = float(va["auc_macro_ovr"]) if np.isfinite(float(va["auc_macro_ovr"])) else float("nan")
        va_acc = float(va["acc"]) if np.isfinite(float(va["acc"])) else float("nan")
        metric = va_auc if np.isfinite(va_auc) else va_acc
        if np.isfinite(va_auc) and va_auc > best_val_auc:
            best_val_auc = va_auc
        if np.isfinite(va_acc) and va_acc > best_val_acc:
            best_val_acc = va_acc

        if np.isfinite(metric) and metric > best_metric:
            best_metric = float(metric)
            best_state_dual = {k: v.detach().cpu().clone() for k, v in dual_model.state_dict().items()}
            best_state_reco = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print_every = 1 if stage_name.startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
            if stagec_mode:
                print(
                    f"{stage_name} ep {ep+1}: train(total/cls/reco/cons/anch/reco_w)="
                    f"{tr_total:.4f}/{tr_cls:.4f}/{tr_reco:.4f}/{tr_cons:.4f}/{tr_anchor:.4f}/{lambda_reco_eff:.4f} | "
                    f"val(loss/acc/auc/fpr50_sigbg/fpr50_ratio)="
                    f"{float(va['loss']):.4f}/{float(va['acc']):.4f}/{float(va['auc_macro_ovr']):.4f}/"
                    f"{float(va['signal_vs_bg_fpr50']):.6f}/{float(va['target_vs_bg_ratio_fpr50']):.6f} "
                    f"best_metric={best_metric:.4f}"
                )
            else:
                print(
                    f"{stage_name} ep {ep+1}: train(total/cls/reco/cons)="
                    f"{tr_total:.4f}/{tr_cls:.4f}/{tr_reco:.4f}/{tr_cons:.4f} | "
                    f"val(loss/acc/auc/fpr50_sigbg/fpr50_ratio)="
                    f"{float(va['loss']):.4f}/{float(va['acc']):.4f}/{float(va['auc_macro_ovr']):.4f}/"
                    f"{float(va['signal_vs_bg_fpr50']):.6f}/{float(va['target_vs_bg_ratio_fpr50']):.6f} "
                    f"best_metric={best_metric:.4f}"
                )

        if (ep + 1) >= int(min_epochs) and wait >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual is not None:
        dual_model.load_state_dict(best_state_dual)
    if best_state_reco is not None:
        reconstructor.load_state_dict(best_state_reco)

    metrics = {
        "best_val_metric": float(best_metric),
        "best_val_auc_macro_ovr": float(best_val_auc),
        "best_val_acc": float(best_val_acc),
        "stagec_progressive_unfreeze": bool(stagec_mode and STAGEC_PROGRESSIVE_UNFREEZE),
        "stagec_unfreeze_phase1_epochs": int(STAGEC_UNFREEZE_PHASE1_EPOCHS),
        "stagec_unfreeze_phase2_epochs": int(STAGEC_UNFREEZE_PHASE2_EPOCHS),
        "stagec_unfreeze_last_n_encoder_layers": int(STAGEC_UNFREEZE_LAST_N_ENCODER_LAYERS),
        "stagec_lambda_param_anchor": float(lambda_param_anchor),
        "stagec_lambda_output_anchor": float(lambda_output_anchor),
        "stagec_anchor_decay": float(anchor_decay),
        "stagec_reco_ramp_epochs": int(STAGEC_RECO_RAMP_EPOCHS),
    }
    states = {"dual": best_state_dual, "reco": best_state_reco}
    return reconstructor, dual_model, metrics, states


def _identity_wrap(model: nn.Module) -> nn.Module:
    return model


def _identity_enforce(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return out


def _patch_pipeline() -> None:
    # HLT style.
    base.build_hlt_view = m2hlt._build_hlt_view_m2style  # type: ignore[assignment]

    # Reconstructor/loss/view.
    base.OfflineReconstructor = jl.OfflineReconstructorJetLatentSet2Set  # type: ignore[assignment]
    base.compute_reconstruction_losses_weighted = jl.compute_reconstruction_losses_weighted_set2set  # type: ignore[assignment]
    base.build_soft_corrected_view = jl.build_soft_corrected_view_set2set  # type: ignore[assignment]
    base.wrap_reconstructor_unmerge_only = _identity_wrap  # type: ignore[assignment]

    # Stage-A trainer lives in reco_joint module and calls its own globals.
    # Patch there too so it uses set2set-compatible loss/behavior.
    reco_joint.compute_reconstruction_losses_weighted = jl.compute_reconstruction_losses_weighted_set2set  # type: ignore[assignment]
    reco_joint.enforce_unmerge_only_output = _identity_enforce  # type: ignore[assignment]
    reco_joint.wrap_reconstructor_unmerge_only = _identity_wrap  # type: ignore[assignment]

    # Stage-C controls.
    base.train_joint_dual_multiclass = train_joint_dual_multiclass_with_stagec_controls  # type: ignore[assignment]


def main() -> None:
    _patch_pipeline()
    args = base.parse_args()
    base.run(args)


if __name__ == "__main__":
    main()
