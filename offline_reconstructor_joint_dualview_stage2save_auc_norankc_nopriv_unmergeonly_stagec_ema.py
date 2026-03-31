#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper entrypoint:
- Reuses full pipeline from
  offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly.py
- Adds Stage-C EMA evaluation/selection:
  * Stage A/B unchanged
  * Stage C trains raw weights
  * Per-step EMA shadows are updated
  * Validation selection uses EMA model outputs
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as base


_ORIG_TRAIN_JOINT_DUAL = base.train_joint_dual

_STAGEC_EMA_ENABLE = int(os.environ.get("STAGEC_EMA_ENABLE", "1")) != 0
_STAGEC_EMA_DECAY = float(max(min(float(os.environ.get("STAGEC_EMA_DECAY", "0.999")), 0.999999), 0.0))
_STAGEC_EMA_WARMUP_EPOCHS = int(max(int(os.environ.get("STAGEC_EMA_WARMUP_EPOCHS", "0")), 0))


@torch.no_grad()
def _ema_update_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_sd = ema_model.state_dict()
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k not in ema_sd:
            continue
        ev = ema_sd[k]
        src = v.detach()
        if torch.is_floating_point(ev):
            ev.mul_(float(decay)).add_(src, alpha=(1.0 - float(decay)))
        else:
            ev.copy_(src)


def train_joint_dual_stagec_ema(
    reconstructor: base.OfflineReconstructor,
    dual_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_dual: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_rank: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    min_epochs: int,
    select_metric: str = "auc",
    apply_cls_weight: bool = False,
    apply_reco_weight: bool = False,
    val_weight_key: Optional[str] = None,
    use_weighted_val_selection: bool = False,
    lambda_delta_cls: float = 0.0,
    delta_tau: float = 0.05,
    delta_lambda_fp: float = 3.0,
    delta_hlt_model: Optional[nn.Module] = None,
    delta_hlt_threshold_prob: float = 0.50,
    delta_warmup_epochs: int = 0,
    progressive_unfreeze: bool = False,
    unfreeze_phase1_epochs: int = 3,
    unfreeze_phase2_epochs: int = 7,
    unfreeze_last_n_encoder_layers: int = 2,
    alternate_freeze: bool = False,
    alternate_reco_only_epochs: int = 5,
    alternate_dual_only_epochs: int = 5,
    lambda_param_anchor: float = 0.0,
    lambda_output_anchor: float = 0.0,
    anchor_decay: float = 1.0,
) -> Tuple[base.OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    use_ema_stagec = bool(
        _STAGEC_EMA_ENABLE
        and str(stage_name).startswith("StageC")
        and (not freeze_reconstructor)
        and (_STAGEC_EMA_DECAY > 0.0)
    )

    # Keep original behavior outside Stage C (or if EMA disabled).
    if not use_ema_stagec:
        return _ORIG_TRAIN_JOINT_DUAL(
            reconstructor=reconstructor,
            dual_model=dual_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            stage_name=stage_name,
            freeze_reconstructor=freeze_reconstructor,
            epochs=epochs,
            patience=patience,
            lr_dual=lr_dual,
            lr_reco=lr_reco,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            lambda_reco=lambda_reco,
            lambda_rank=lambda_rank,
            lambda_cons=lambda_cons,
            corrected_weight_floor=corrected_weight_floor,
            corrected_use_flags=corrected_use_flags,
            min_epochs=min_epochs,
            select_metric=select_metric,
            apply_cls_weight=apply_cls_weight,
            apply_reco_weight=apply_reco_weight,
            val_weight_key=val_weight_key,
            use_weighted_val_selection=use_weighted_val_selection,
            lambda_delta_cls=lambda_delta_cls,
            delta_tau=delta_tau,
            delta_lambda_fp=delta_lambda_fp,
            delta_hlt_model=delta_hlt_model,
            delta_hlt_threshold_prob=delta_hlt_threshold_prob,
            delta_warmup_epochs=delta_warmup_epochs,
            progressive_unfreeze=progressive_unfreeze,
            unfreeze_phase1_epochs=unfreeze_phase1_epochs,
            unfreeze_phase2_epochs=unfreeze_phase2_epochs,
            unfreeze_last_n_encoder_layers=unfreeze_last_n_encoder_layers,
            alternate_freeze=alternate_freeze,
            alternate_reco_only_epochs=alternate_reco_only_epochs,
            alternate_dual_only_epochs=alternate_dual_only_epochs,
            lambda_param_anchor=lambda_param_anchor,
            lambda_output_anchor=lambda_output_anchor,
            anchor_decay=anchor_decay,
        )

    def _set_reco_trainability(mode: str) -> int:
        for p in reconstructor.parameters():
            p.requires_grad_(False)

        if mode == "frozen":
            return 0

        if mode == "all":
            for p in reconstructor.parameters():
                p.requires_grad_(True)
            return sum(int(p.numel()) for p in reconstructor.parameters())

        trainable_prefixes: List[str] = [
            "token_norm",
            "action_head",
            "unsmear_head",
            "reassign_head",
            "split_exist_head",
            "split_delta_head",
            "pool_attn",
            "budget_head",
            "gen_attn",
            "gen_norm",
            "gen_head",
            "gen_exist_head",
        ]
        if mode == "lastk":
            n_layers = len(reconstructor.encoder_layers)
            k = int(max(0, min(int(unfreeze_last_n_encoder_layers), n_layers)))
            for idx in range(max(0, n_layers - k), n_layers):
                trainable_prefixes.append(f"encoder_layers.{idx}")

        trainable = 0
        for name, p in reconstructor.named_parameters():
            if any(name.startswith(pref) for pref in trainable_prefixes):
                p.requires_grad_(True)
                trainable += int(p.numel())
        return trainable

    def _set_dual_trainability(trainable: bool) -> int:
        n = 0
        for p in dual_model.parameters():
            p.requires_grad_(bool(trainable))
            if bool(trainable):
                n += int(p.numel())
        return n

    for p in reconstructor.parameters():
        p.requires_grad_(True)
    for p in dual_model.parameters():
        p.requires_grad_(True)

    params = [
        {"params": dual_model.parameters(), "lr": float(lr_dual)},
        {"params": reconstructor.parameters(), "lr": float(lr_reco)},
    ]
    opt = torch.optim.AdamW(params, lr=float(lr_dual), weight_decay=float(weight_decay))
    sch = base.get_scheduler(opt, int(warmup_epochs), int(epochs))

    # EMA shadows used for validation/selection only.
    ema_reconstructor = copy.deepcopy(reconstructor).to(device)
    ema_dual = copy.deepcopy(dual_model).to(device)
    ema_reconstructor.eval()
    ema_dual.eval()
    for p in ema_reconstructor.parameters():
        p.requires_grad_(False)
    for p in ema_dual.parameters():
        p.requires_grad_(False)

    best_state_dual_sel = None
    best_state_reco_sel = None
    best_state_dual_auc = None
    best_state_reco_auc = None
    best_state_dual_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_val_fpr50_unw = float("inf")
    best_val_auc_unw = float("-inf")
    best_val_fpr50_w = float("inf")
    best_val_auc_w = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    sel_val_fpr50_unw = float("nan")
    sel_val_auc_unw = float("nan")
    sel_val_fpr50_w = float("nan")
    sel_val_auc_w = float("nan")
    val_metric_source = "weighted" if bool(use_weighted_val_selection) else "unweighted"
    no_improve = 0

    delta_tau = float(max(delta_tau, 1e-6))
    delta_lambda_fp = float(max(delta_lambda_fp, 0.0))
    lambda_delta_cls = float(max(lambda_delta_cls, 0.0))
    use_delta = bool(lambda_delta_cls > 0.0 and delta_hlt_model is not None)
    if use_delta:
        delta_hlt_model.eval()
        for p in delta_hlt_model.parameters():
            p.requires_grad_(False)

    lambda_param_anchor = float(max(lambda_param_anchor, 0.0))
    lambda_output_anchor = float(max(lambda_output_anchor, 0.0))
    anchor_decay = float(max(anchor_decay, 0.0))
    use_anchor = bool(lambda_param_anchor > 0.0 or lambda_output_anchor > 0.0)
    reco_anchor_copy = None
    reco_anchor_state = None
    if use_anchor:
        reco_anchor_copy = copy.deepcopy(reconstructor).to(device)
        reco_anchor_copy.eval()
        for p in reco_anchor_copy.parameters():
            p.requires_grad_(False)
        reco_anchor_state = {
            k: v.detach().clone().to(device)
            for k, v in reconstructor.state_dict().items()
            if torch.is_tensor(v)
        }

    alt_reco_epochs_i = int(max(alternate_reco_only_epochs, 0))
    alt_dual_epochs_i = int(max(alternate_dual_only_epochs, 0))
    alt_cycle_len = int(alt_reco_epochs_i + alt_dual_epochs_i)
    alt_schedule_enabled = bool(
        bool(alternate_freeze)
        and alt_reco_epochs_i > 0
        and alt_dual_epochs_i > 0
        and alt_cycle_len > 0
    )
    progressive_unfreeze_active = bool(progressive_unfreeze) and (not alt_schedule_enabled)
    if alt_schedule_enabled and bool(progressive_unfreeze):
        print(
            "Note: Stage-C alternating freeze is enabled; "
            "ignoring --stageC_progressive_unfreeze schedule."
        )

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        current_reco_frozen = False
        current_dual_frozen = False

        if alt_schedule_enabled:
            cycle_pos = int(ep % alt_cycle_len)
            if cycle_pos < alt_reco_epochs_i:
                dual_model.eval()
                n_trainable_dual = _set_dual_trainability(False)
                reconstructor.train()
                n_trainable_reco = _set_reco_trainability("all")
                current_unfreeze_mode = "alt_reco_only"
                current_reco_frozen = False
                current_dual_frozen = True
            else:
                dual_model.train()
                n_trainable_dual = _set_dual_trainability(True)
                reconstructor.eval()
                n_trainable_reco = _set_reco_trainability("frozen")
                current_unfreeze_mode = "alt_dual_only"
                current_reco_frozen = True
                current_dual_frozen = False
        else:
            dual_model.train()
            n_trainable_dual = _set_dual_trainability(True)
            reconstructor.train()
            current_reco_frozen = False
            if bool(progressive_unfreeze_active):
                if (ep + 1) <= int(max(unfreeze_phase1_epochs, 0)):
                    current_unfreeze_mode = "heads"
                elif (ep + 1) <= int(max(unfreeze_phase2_epochs, 0)):
                    current_unfreeze_mode = "lastk"
                else:
                    current_unfreeze_mode = "all"
                n_trainable_reco = _set_reco_trainability(current_unfreeze_mode)
            else:
                current_unfreeze_mode = "all"
                n_trainable_reco = _set_reco_trainability("all")

        tr_loss = tr_cls = tr_rank = tr_reco = tr_cons = 0.0
        tr_delta = tr_delta_gain = tr_delta_cost = 0.0
        tr_anchor_param = tr_anchor_out = 0.0
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
            sw_cls = batch.get("sample_weight_cls", None)
            sw_reco = batch.get("sample_weight_reco", None)
            if sw_cls is not None:
                sw_cls = sw_cls.to(device)
            if sw_reco is not None:
                sw_reco = sw_reco.to(device)

            opt.zero_grad()

            if current_reco_frozen:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
                reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)

            feat_b, mask_b = base.build_soft_corrected_view(
                reco_out,
                weight_floor=corrected_weight_floor,
                scale_features_by_weight=True,
                include_flags=corrected_use_flags,
            )
            logits = dual_model(feat_hlt_dual, mask_hlt, feat_b, mask_b).squeeze(1)

            if bool(apply_cls_weight) and sw_cls is not None:
                loss_cls_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                denom = sw_cls.sum().clamp(min=1e-6)
                loss_cls = (loss_cls_raw * sw_cls).sum() / denom
            else:
                loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = base.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            loss_delta = torch.zeros((), device=device)
            delta_gain = torch.zeros((), device=device)
            delta_cost = torch.zeros((), device=device)
            if use_delta:
                with torch.no_grad():
                    logits_hlt = delta_hlt_model(feat_hlt_reco, mask_hlt).squeeze(1)
                    p_hlt = torch.sigmoid(logits_hlt)
                    h_soft = torch.sigmoid((p_hlt - float(delta_hlt_threshold_prob)) / float(delta_tau))
                y_f = y.float().view(-1)
                p_joint = torch.sigmoid(logits).view(-1)
                miss_hlt = (1.0 - h_soft).view(-1)
                delta_gain = (y_f * miss_hlt * p_joint).mean()
                delta_cost = ((1.0 - y_f) * miss_hlt * p_joint).mean()
                loss_delta = -delta_gain + float(delta_lambda_fp) * delta_cost
                warm = int(max(delta_warmup_epochs, 0))
                if warm > 0:
                    ramp = min(1.0, float(ep + 1) / float(warm))
                else:
                    ramp = 1.0
            else:
                ramp = 1.0

            if float(lambda_reco) > 0.0:
                reco_losses = base.compute_reconstruction_losses_weighted(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    base.BASE_CONFIG["loss"],
                    sample_weight=(sw_reco if (bool(apply_reco_weight) and sw_reco is not None) else None),
                )
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            if use_anchor:
                if reco_anchor_copy is None or reco_anchor_state is None:
                    raise RuntimeError("Anchor mode enabled but anchor references are missing.")
                with torch.no_grad():
                    out_anchor = reco_anchor_copy(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
                loss_anchor_out = (
                    F.mse_loss(reco_out["cand_weights"], out_anchor["cand_weights"])
                    + F.mse_loss(reco_out["budget_total"], out_anchor["budget_total"])
                    + F.mse_loss(reco_out["budget_merge"], out_anchor["budget_merge"])
                    + F.mse_loss(reco_out["budget_eff"], out_anchor["budget_eff"])
                )
                loss_anchor_param = torch.zeros((), device=device)
                if lambda_param_anchor > 0.0:
                    for name, p in reconstructor.named_parameters():
                        if p.requires_grad:
                            ref = reco_anchor_state[name]
                            loss_anchor_param = loss_anchor_param + F.mse_loss(p, ref)
                anchor_scale = float(anchor_decay) ** float(ep)
            else:
                loss_anchor_out = torch.zeros((), device=device)
                loss_anchor_param = torch.zeros((), device=device)
                anchor_scale = 1.0

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
                + float(lambda_delta_cls) * float(ramp) * loss_delta
                + float(anchor_scale) * float(lambda_param_anchor) * loss_anchor_param
                + float(anchor_scale) * float(lambda_output_anchor) * loss_anchor_out
            )
            loss.backward()
            if not current_dual_frozen:
                torch.nn.utils.clip_grad_norm_(dual_model.parameters(), 1.0)
            if not current_reco_frozen:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            if (ep + 1) > int(_STAGEC_EMA_WARMUP_EPOCHS):
                _ema_update_model(ema_dual, dual_model, _STAGEC_EMA_DECAY)
                _ema_update_model(ema_reconstructor, reconstructor, _STAGEC_EMA_DECAY)

            bs = feat_hlt_reco.size(0)
            tr_loss += loss.item() * bs
            tr_cls += loss_cls.item() * bs
            tr_rank += loss_rank.item() * bs
            tr_reco += loss_reco.item() * bs
            tr_cons += loss_cons.item() * bs
            tr_delta += loss_delta.item() * bs
            tr_delta_gain += delta_gain.item() * bs
            tr_delta_cost += delta_cost.item() * bs
            tr_anchor_param += loss_anchor_param.item() * bs
            tr_anchor_out += loss_anchor_out.item() * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)
        tr_delta /= max(n_tr, 1)
        tr_delta_gain /= max(n_tr, 1)
        tr_delta_cost /= max(n_tr, 1)
        tr_anchor_param /= max(n_tr, 1)
        tr_anchor_out /= max(n_tr, 1)

        eval_reco = ema_reconstructor
        eval_dual = ema_dual
        va_pack = base.eval_joint_model_both_metrics(
            reconstructor=eval_reco,
            dual_model=eval_dual,
            loader=val_loader,
            device=device,
            corrected_weight_floor=corrected_weight_floor,
            corrected_use_flags=corrected_use_flags,
            weighted_key=val_weight_key if bool(use_weighted_val_selection) else None,
        )
        va_auc_unw = float(va_pack["auc_unweighted"])
        va_fpr50_unw = float(va_pack["fpr50_unweighted"])
        va_auc_w = float(va_pack["auc_weighted"])
        va_fpr50_w = float(va_pack["fpr50_weighted"])
        has_weighted_val = bool(use_weighted_val_selection) and np.isfinite(va_auc_w) and np.isfinite(va_fpr50_w)
        va_auc = float(va_auc_w) if has_weighted_val else float(va_auc_unw)
        va_fpr50 = float(va_fpr50_w) if has_weighted_val else float(va_fpr50_unw)
        metric_source_epoch = "weighted" if has_weighted_val else "unweighted"

        if np.isfinite(va_fpr50_unw) and float(va_fpr50_unw) < best_val_fpr50_unw:
            best_val_fpr50_unw = float(va_fpr50_unw)
        if np.isfinite(va_auc_unw) and float(va_auc_unw) > best_val_auc_unw:
            best_val_auc_unw = float(va_auc_unw)
        if np.isfinite(va_fpr50_w) and float(va_fpr50_w) < best_val_fpr50_w:
            best_val_fpr50_w = float(va_fpr50_w)
        if np.isfinite(va_auc_w) and float(va_auc_w) > best_val_auc_w:
            best_val_auc_w = float(va_auc_w)

        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_dual_fpr = {k: v.detach().cpu().clone() for k, v in eval_dual.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in eval_reco.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_dual_auc = {k: v.detach().cpu().clone() for k, v in eval_dual.state_dict().items()}
            best_state_reco_auc = {k: v.detach().cpu().clone() for k, v in eval_reco.state_dict().items()}

        if str(select_metric).lower() == "auc":
            improved = np.isfinite(va_auc) and (float(va_auc) > best_sel_score)
            current_score = float(va_auc) if np.isfinite(va_auc) else float("-inf")
        else:
            improved = np.isfinite(va_fpr50) and (float(va_fpr50) < best_sel_score)
            current_score = float(va_fpr50) if np.isfinite(va_fpr50) else float("inf")

        if improved:
            best_sel_score = current_score
            sel_val_fpr50 = float(va_fpr50)
            sel_val_auc = float(va_auc)
            sel_val_fpr50_unw = float(va_fpr50_unw)
            sel_val_auc_unw = float(va_auc_unw)
            sel_val_fpr50_w = float(va_fpr50_w)
            sel_val_auc_w = float(va_auc_w)
            val_metric_source = str(metric_source_epoch)
            best_state_dual_sel = {k: v.detach().cpu().clone() for k, v in eval_dual.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in eval_reco.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print_every = 1
        if (ep + 1) % print_every == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}, "
                f"delta={tr_delta:.4f}, d_gain={tr_delta_gain:.4f}, d_cost={tr_delta_cost:.4f}, "
                f"anc_p={tr_anchor_param:.4f}, anc_o={tr_anchor_out:.4f}) | "
                f"val_auc_unw={va_auc_unw:.4f}, val_fpr50_unw={va_fpr50_unw:.6f}, "
                f"val_auc_w={va_auc_w:.4f}, val_fpr50_w={va_fpr50_w:.6f}, "
                f"val_metric_source={metric_source_epoch}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}, "
                f"unfreeze={current_unfreeze_mode}, reco_trainable={n_trainable_reco}, "
                f"dual_frozen={current_dual_frozen}, dual_trainable={n_trainable_dual}, "
                f"ema_eval=1, ema_decay={_STAGEC_EMA_DECAY:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_dual_sel is not None:
        dual_model.load_state_dict(best_state_dual_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "val_metric_source": str(val_metric_source),
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "selected_val_fpr50_unweighted": float(sel_val_fpr50_unw),
        "selected_val_auc_unweighted": float(sel_val_auc_unw),
        "selected_val_fpr50_weighted": float(sel_val_fpr50_w),
        "selected_val_auc_weighted": float(sel_val_auc_w),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
        "best_val_fpr50_seen_unweighted": float(best_val_fpr50_unw),
        "best_val_auc_seen_unweighted": float(best_val_auc_unw),
        "best_val_fpr50_seen_weighted": float(best_val_fpr50_w),
        "best_val_auc_seen_weighted": float(best_val_auc_w),
        "delta_enabled": bool(use_delta),
        "delta_lambda": float(lambda_delta_cls),
        "delta_tau": float(delta_tau),
        "delta_lambda_fp": float(delta_lambda_fp),
        "delta_hlt_threshold_prob": float(delta_hlt_threshold_prob),
        "delta_warmup_epochs": int(max(delta_warmup_epochs, 0)),
        "progressive_unfreeze": bool(progressive_unfreeze_active),
        "unfreeze_phase1_epochs": int(unfreeze_phase1_epochs),
        "unfreeze_phase2_epochs": int(unfreeze_phase2_epochs),
        "unfreeze_last_n_encoder_layers": int(unfreeze_last_n_encoder_layers),
        "alternate_freeze": bool(alt_schedule_enabled),
        "alternate_reco_only_epochs": int(alt_reco_epochs_i),
        "alternate_dual_only_epochs": int(alt_dual_epochs_i),
        "lambda_param_anchor": float(lambda_param_anchor),
        "lambda_output_anchor": float(lambda_output_anchor),
        "anchor_decay": float(anchor_decay),
        "stagec_ema_enabled": True,
        "stagec_ema_decay": float(_STAGEC_EMA_DECAY),
        "stagec_ema_warmup_epochs": int(_STAGEC_EMA_WARMUP_EPOCHS),
    }
    state_pack = {
        "selected": {"dual": best_state_dual_sel, "reco": best_state_reco_sel},
        "auc": {"dual": best_state_dual_auc, "reco": best_state_reco_auc},
        "fpr50": {"dual": best_state_dual_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, dual_model, metrics, state_pack


base.train_joint_dual = train_joint_dual_stagec_ema


if __name__ == "__main__":
    print(
        "[StageC-EMA] enabled "
        f"(decay={_STAGEC_EMA_DECAY:.6f}, warmup_epochs={_STAGEC_EMA_WARMUP_EPOCHS})"
    )
    base.main()

