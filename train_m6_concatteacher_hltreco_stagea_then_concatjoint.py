#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as b
import reco_concat_teacher_stageA_then_corrected as m6


def _deepcopy_cfg() -> Dict:
    return b._deepcopy_config()


def _build_concat_constituents_torch(
    const_a: torch.Tensor,
    mask_a: torch.Tensor,
    const_b: torch.Tensor,
    mask_b: torch.Tensor,
    max_concat_constits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate two constituent sets along token axis and truncate/pad to max_concat_constits.
    """
    const_cat = torch.cat([const_a, const_b], dim=1)
    mask_cat = torch.cat([mask_a.bool(), mask_b.bool()], dim=1)

    full_l = int(const_cat.shape[1])
    out_l = int(max_concat_constits)
    if out_l <= 0:
        out_l = full_l

    if out_l < full_l:
        const_cat = const_cat[:, :out_l, :]
        mask_cat = mask_cat[:, :out_l]
    elif out_l > full_l:
        n = int(const_cat.shape[0])
        d = int(const_cat.shape[2])
        pad_const = const_cat.new_zeros((n, out_l - full_l, d))
        pad_mask = torch.zeros((n, out_l - full_l), dtype=torch.bool, device=mask_cat.device)
        const_cat = torch.cat([const_cat, pad_const], dim=1)
        mask_cat = torch.cat([mask_cat, pad_mask], dim=1)

    const_cat = torch.where(mask_cat.unsqueeze(-1), const_cat, torch.zeros_like(const_cat))
    return const_cat, mask_cat


def _build_concat_view_numpy(
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    feat_corr: np.ndarray,
    mask_corr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build classifier single-view features by concatenating token streams:
    [HLT tokens] + [soft-corrected tokens], with a source-flag channel.
    """
    bsz, lh, dh = feat_hlt.shape
    _, lc, dc = feat_corr.shape
    d_base = max(int(dh), int(dc))

    h_pad = np.zeros((bsz, lh, d_base), dtype=np.float32)
    c_pad = np.zeros((bsz, lc, d_base), dtype=np.float32)
    h_pad[:, :, :dh] = feat_hlt.astype(np.float32)
    c_pad[:, :, :dc] = feat_corr.astype(np.float32)

    h_src = np.zeros((bsz, lh, 1), dtype=np.float32)
    c_src = np.ones((bsz, lc, 1), dtype=np.float32)
    h_aug = np.concatenate([h_pad, h_src], axis=-1)
    c_aug = np.concatenate([c_pad, c_src], axis=-1)

    feat_cat = np.concatenate([h_aug, c_aug], axis=1)
    mask_cat = np.concatenate([mask_hlt.astype(bool), mask_corr.astype(bool)], axis=1)
    feat_cat *= mask_cat[..., None].astype(np.float32)
    return feat_cat.astype(np.float32), mask_cat.astype(bool)


def _build_concat_view_torch(
    feat_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    feat_corr: torch.Tensor,
    mask_corr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, lh, dh = feat_hlt.shape
    _, lc, dc = feat_corr.shape
    d_base = max(int(dh), int(dc))

    h_pad = feat_hlt.new_zeros((bsz, lh, d_base))
    c_pad = feat_hlt.new_zeros((bsz, lc, d_base))
    h_pad[:, :, :dh] = feat_hlt
    c_pad[:, :, :dc] = feat_corr

    h_src = feat_hlt.new_zeros((bsz, lh, 1))
    c_src = feat_hlt.new_ones((bsz, lc, 1))
    h_aug = torch.cat([h_pad, h_src], dim=-1)
    c_aug = torch.cat([c_pad, c_src], dim=-1)

    feat_cat = torch.cat([h_aug, c_aug], dim=1)
    mask_cat = torch.cat([mask_hlt.bool(), mask_corr.bool()], dim=1)
    feat_cat = feat_cat * mask_cat.unsqueeze(-1).float()
    return feat_cat, mask_cat


def _compute_concat_teacher_guided_reco_losses_hltreco(
    reco_out: Dict[str, torch.Tensor],
    const_hlt: torch.Tensor,
    mask_hlt: torch.Tensor,
    const_off: torch.Tensor,
    mask_off: torch.Tensor,
    const_teacher: torch.Tensor,
    mask_teacher: torch.Tensor,
    budget_merge_true: torch.Tensor,
    budget_eff_true: torch.Tensor,
    teacher_model: torch.nn.Module,
    means_t: torch.Tensor,
    stds_t: torch.Tensor,
    loss_cfg: Dict,
    kd_temperature: float,
    budget_eps: float,
    budget_weight_floor: float,
) -> Dict[str, torch.Tensor]:
    """
    Modified Stage-A teacher-guided loss:
      target path: teacher(concat(HLT, offline))
      pred path:   teacher(concat(HLT, reco_pred))
    Physics/budget losses remain anchored to offline.
    """
    aux_losses = b.compute_reconstruction_losses(
        reco_out,
        const_hlt,
        mask_hlt,
        const_off,
        mask_off,
        budget_merge_true,
        budget_eff_true,
        loss_cfg,
    )
    loss_phys = aux_losses["phys"]

    with torch.no_grad():
        # target teacher signal on concat(HLT, offline)
        feat_teacher_raw = b.compute_features_torch(const_teacher, mask_teacher)
        feat_teacher_std = m6.b._standardize_features_torch(feat_teacher_raw, mask_teacher, means_t, stds_t)
        teacher_pack = teacher_model(feat_teacher_std, mask_teacher, return_attention=True, return_embedding=True)
        logits_teacher_target = teacher_pack[0].view(-1)
        attn_teacher_target = teacher_pack[1]
        emb_teacher_target = teacher_pack[2]

    # reco tokens derived from reconstructor output.
    feat_reco_tok, mask_reco_tok = m6.b._build_teacher_reco_features_from_output(
        reco_out,
        const_hlt,
        mask_hlt,
        weight_floor=budget_weight_floor,
    )
    # Convert reco features back to constituent-like tokens by reading tok branch from reco_out.
    # _build_teacher_reco_features_from_output uses tok tokens and mask; we reconstruct same token set.
    l_tok = int(const_hlt.shape[1])
    reco_tok_const = reco_out["cand_tokens"][:, :l_tok, :]
    reco_tok_mask = mask_reco_tok

    # Build concat(HLT, reco_pred) with same max length as target concat.
    max_concat_constits = int(const_teacher.shape[1])
    const_pred_cat, mask_pred_cat = _build_concat_constituents_torch(
        const_hlt,
        mask_hlt,
        reco_tok_const,
        reco_tok_mask,
        max_concat_constits=max_concat_constits,
    )
    feat_pred_raw = b.compute_features_torch(const_pred_cat, mask_pred_cat)
    feat_pred_std = m6.b._standardize_features_torch(feat_pred_raw, mask_pred_cat, means_t, stds_t)
    reco_pack = teacher_model(feat_pred_std, mask_pred_cat, return_attention=True, return_embedding=True)
    logits_teacher_reco = reco_pack[0].view(-1)
    attn_teacher_reco = reco_pack[1]
    emb_teacher_reco = reco_pack[2]

    kd_temperature = max(float(kd_temperature), 1e-3)
    target_soft = torch.sigmoid(logits_teacher_target / kd_temperature)
    kd_vec = (
        F.binary_cross_entropy_with_logits(
            logits_teacher_reco / kd_temperature,
            target_soft,
            reduction="none",
        )
        * (kd_temperature * kd_temperature)
    )
    loss_kd = m6.b._weighted_batch_mean(kd_vec, None)

    emb_target_n = F.normalize(emb_teacher_target, dim=1)
    emb_reco_n = F.normalize(emb_teacher_reco, dim=1)
    loss_emb = (1.0 - (emb_target_n * emb_reco_n).sum(dim=1)).mean()

    def _attn_to_token_vec(attn: torch.Tensor, l_take: int) -> torch.Tensor:
        if attn.dim() == 2:
            return attn[:, :l_take]
        if attn.dim() == 3:
            return attn[:, :l_take, :l_take].mean(dim=1)
        if attn.dim() == 4:
            return attn[:, :, :l_take, :l_take].mean(dim=(1, 2))
        raise RuntimeError(f"Unexpected attention rank={attn.dim()} shape={tuple(attn.shape)}")

    l_reco = int(mask_pred_cat.shape[1])
    l_teacher = int(mask_teacher.shape[1])
    l_common = int(min(l_reco, l_teacher))
    if l_common > 0:
        attn_pred_tok = _attn_to_token_vec(attn_teacher_reco, l_common)
        attn_tgt_tok = _attn_to_token_vec(attn_teacher_target, l_common)
        mask_pred_tok = mask_pred_cat[:, :l_common]
        mask_tgt_tok = mask_teacher[:, :l_common]
        loss_tok = m6.b._attention_kl_loss_masked(
            attn_pred=attn_pred_tok,
            attn_target=attn_tgt_tok,
            mask_pred=mask_pred_tok,
            mask_target=mask_tgt_tok,
        )
    else:
        loss_tok = torch.zeros((), device=mask_pred_cat.device)

    reco_tokens = reco_out["cand_tokens"][:, : const_hlt.shape[1], :]
    mean_edit_vec = m6.b._sorted_edit_budget_vec(reco_tokens, const_hlt, mask_hlt)
    budget_hinge_vec = F.relu(mean_edit_vec - float(budget_eps))
    loss_budget_hinge = m6.b._weighted_batch_mean(budget_hinge_vec, None)

    return {
        "kd": loss_kd,
        "emb": loss_emb,
        "tok": loss_tok,
        "phys": loss_phys,
        "budget_hinge": loss_budget_hinge,
        "logits_teacher_reco": logits_teacher_reco,
    }


@torch.no_grad()
def eval_teacher_on_concat_hltreco_split(
    reconstructor: torch.nn.Module,
    teacher: torch.nn.Module,
    feat_hlt_std: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_const: np.ndarray,
    labels: np.ndarray,
    split_idx: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    weight_floor: float,
    max_concat_constits: int,
    target_tpr: float,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    reconstructor.eval()
    teacher.eval()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(stds, 1e-6, None), dtype=torch.float32, device=device)

    preds_list: List[np.ndarray] = []
    labs_list: List[np.ndarray] = []

    idx = split_idx.astype(np.int64)
    for start in range(0, len(idx), int(batch_size)):
        end = min(start + int(batch_size), len(idx))
        sl = idx[start:end]
        x = torch.tensor(feat_hlt_std[sl], dtype=torch.float32, device=device)
        m = torch.tensor(hlt_mask[sl], dtype=torch.bool, device=device)
        c = torch.tensor(hlt_const[sl], dtype=torch.float32, device=device)

        reco_out = reconstructor(x, m, c, stage_scale=1.0)
        _feat_reco, mask_reco = m6.b._build_teacher_reco_features_from_output(
            reco_out, c, m, weight_floor=float(weight_floor)
        )
        l_tok = int(c.shape[1])
        reco_tok_const = reco_out["cand_tokens"][:, :l_tok, :]
        c_cat, m_cat = _build_concat_constituents_torch(c, m, reco_tok_const, mask_reco, int(max_concat_constits))
        feat_cat = b.compute_features_torch(c_cat, m_cat)
        feat_cat_std = m6.b._standardize_features_torch(feat_cat, m_cat, means_t, stds_t)
        logits = teacher(feat_cat_std, m_cat).squeeze(1)
        p = torch.sigmoid(logits)

        preds_list.append(p.detach().cpu().numpy().astype(np.float64))
        labs_list.append(labels[sl].astype(np.float32))

    preds = np.concatenate(preds_list) if preds_list else np.zeros(0, dtype=np.float64)
    labs = np.concatenate(labs_list) if labs_list else np.zeros(0, dtype=np.float32)

    auc = float(roc_auc_score(labs, preds)) if len(np.unique(labs)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labs, preds)
    fpr_at = float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))
    return auc, preds, labs, fpr_at


class ConcatJointDataset(Dataset):
    def __init__(
        self,
        feat_hlt_reco: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt: np.ndarray,
        const_off: np.ndarray,
        mask_off: np.ndarray,
        budget_merge_true: np.ndarray,
        budget_eff_true: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt_reco = torch.tensor(feat_hlt_reco, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt = torch.tensor(const_hlt, dtype=torch.float32)
        self.const_off = torch.tensor(const_off, dtype=torch.float32)
        self.mask_off = torch.tensor(mask_off, dtype=torch.bool)
        self.budget_merge_true = torch.tensor(budget_merge_true, dtype=torch.float32)
        self.budget_eff_true = torch.tensor(budget_eff_true, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt_reco": self.feat_hlt_reco[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt": self.const_hlt[i],
            "const_off": self.const_off[i],
            "mask_off": self.mask_off[i],
            "budget_merge_true": self.budget_merge_true[i],
            "budget_eff_true": self.budget_eff_true[i],
            "label": self.labels[i],
        }


@torch.no_grad()
def eval_concat_joint_model(
    reconstructor: b.OfflineReconstructor,
    concat_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    concat_model.eval()
    reconstructor.eval()

    preds = []
    labs = []
    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].to(device)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_corr, mask_corr = b.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        feat_cat, mask_cat = _build_concat_view_torch(feat_hlt_reco, mask_hlt, feat_corr, mask_corr)
        logits = concat_model(feat_cat, mask_cat).squeeze(1)
        p = torch.sigmoid(logits)
        preds.append(p.detach().cpu().numpy().astype(np.float64))
        labs.append(y.detach().cpu().numpy().astype(np.float32))

    preds_np = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float64)
    labs_np = np.concatenate(labs) if labs else np.zeros(0, dtype=np.float32)
    if preds_np.size == 0:
        return float("nan"), preds_np, labs_np, float("nan")
    auc = float(roc_auc_score(labs_np, preds_np)) if len(np.unique(labs_np)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(labs_np, preds_np)
    fpr50 = float(b.fpr_at_target_tpr(fpr, tpr, 0.50))
    return auc, preds_np, labs_np, fpr50


def train_concat_joint(
    reconstructor: b.OfflineReconstructor,
    concat_model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    epochs: int,
    patience: int,
    min_epochs: int,
    lr_model: float,
    lr_reco: float,
    weight_decay: float,
    warmup_epochs: int,
    lambda_reco: float,
    lambda_rank: float,
    lambda_cons: float,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    select_metric: str,
) -> Tuple[b.OfflineReconstructor, torch.nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = True

    params = [
        {"params": concat_model.parameters(), "lr": float(lr_model)},
        {"params": reconstructor.parameters(), "lr": float(lr_reco)},
    ]
    opt = torch.optim.AdamW(params, lr=float(lr_model), weight_decay=float(weight_decay))
    sch = b.get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_model_sel = None
    best_state_reco_sel = None
    best_state_model_auc = None
    best_state_reco_auc = None
    best_state_model_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    no_improve = 0

    for ep in range(int(epochs)):
        concat_model.train()
        reconstructor.train()

        tr_loss = tr_cls = tr_rank = tr_reco = tr_cons = 0.0
        n_tr = 0

        for batch in train_loader:
            feat_hlt_reco = batch["feat_hlt_reco"].to(device)
            mask_hlt = batch["mask_hlt"].to(device)
            const_hlt = batch["const_hlt"].to(device)
            const_off = batch["const_off"].to(device)
            mask_off = batch["mask_off"].to(device)
            b_merge = batch["budget_merge_true"].to(device)
            b_eff = batch["budget_eff_true"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()

            reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            feat_corr, mask_corr = b.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            feat_cat, mask_cat = _build_concat_view_torch(feat_hlt_reco, mask_hlt, feat_corr, mask_corr)
            logits = concat_model(feat_cat, mask_cat).squeeze(1)

            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = b.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()
            if float(lambda_reco) > 0.0:
                reco_losses = b.compute_reconstruction_losses(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    b.BASE_CONFIG["loss"],
                )
                loss_reco = reco_losses["total"]
            else:
                loss_reco = torch.zeros((), device=device)

            loss = (
                loss_cls
                + float(lambda_rank) * loss_rank
                + float(lambda_reco) * loss_reco
                + float(lambda_cons) * loss_cons
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(concat_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = feat_hlt_reco.size(0)
            tr_loss += float(loss.item()) * bs
            tr_cls += float(loss_cls.item()) * bs
            tr_rank += float(loss_rank.item()) * bs
            tr_reco += float(loss_reco.item()) * bs
            tr_cons += float(loss_cons.item()) * bs
            n_tr += bs

        sch.step()

        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_rank /= max(n_tr, 1)
        tr_reco /= max(n_tr, 1)
        tr_cons /= max(n_tr, 1)

        va_auc, _, _, va_fpr50 = eval_concat_joint_model(
            reconstructor=reconstructor,
            concat_model=concat_model,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            corrected_use_flags=bool(corrected_use_flags),
        )

        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_model_fpr = {k: v.detach().cpu().clone() for k, v in concat_model.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_model_auc = {k: v.detach().cpu().clone() for k, v in concat_model.state_dict().items()}
            best_state_reco_auc = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}

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
            best_state_model_sel = {k: v.detach().cpu().clone() for k, v in concat_model.state_dict().items()}
            best_state_reco_sel = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 1 == 0:
            print(
                f"{stage_name} ep {ep+1}: train_loss={tr_loss:.4f} "
                f"(cls={tr_cls:.4f}, rank={tr_rank:.4f}, reco={tr_reco:.4f}, cons={tr_cons:.4f}) | "
                f"val_auc={va_auc:.4f}, val_fpr50={va_fpr50:.6f}, "
                f"select={str(select_metric).lower()}, best_sel={best_sel_score:.6f}"
            )

        if (ep + 1) >= int(min_epochs) and no_improve >= int(patience):
            print(f"Early stopping {stage_name} at epoch {ep+1}")
            break

    if best_state_model_sel is not None:
        concat_model.load_state_dict(best_state_model_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
    }
    states = {
        "selected": {"model": best_state_model_sel, "reco": best_state_reco_sel},
        "auc": {"model": best_state_model_auc, "reco": best_state_reco_auc},
        "fpr50": {"model": best_state_model_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, concat_model, metrics, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--n_train_jets", type=int, default=375000)
    ap.add_argument("--offset_jets", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--max_concat_constits", type=int, default=200)
    ap.add_argument("--n_train_split", type=int, default=150000)
    ap.add_argument("--n_val_split", type=int, default=75000)
    ap.add_argument("--n_test_split", type=int, default=150000)
    ap.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model6_concat_hltreco_stagea_concatjoint")
    ap.add_argument("--run_name", type=str, default="model6_concat_hltreco_stagea_concatjoint_150k75k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=b.RANDOM_SEED)
    ap.add_argument("--skip_save_models", action="store_true")

    # HLT effects
    ap.add_argument("--merge_radius", type=float, default=b.BASE_CONFIG["hlt_effects"]["merge_radius"])
    ap.add_argument("--eff_plateau_barrel", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_barrel"])
    ap.add_argument("--eff_plateau_endcap", type=float, default=b.BASE_CONFIG["hlt_effects"]["eff_plateau_endcap"])
    ap.add_argument("--smear_a", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_a"])
    ap.add_argument("--smear_b", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_b"])
    ap.add_argument("--smear_c", type=float, default=b.BASE_CONFIG["hlt_effects"]["smear_c"])

    # Stage-A
    ap.add_argument("--stageA_epochs", type=int, default=90)
    ap.add_argument("--stageA_patience", type=int, default=18)
    ap.add_argument("--stageA_phase035_epochs", type=int, default=-1)
    ap.add_argument("--stageA_phase070_epochs", type=int, default=-1)
    ap.add_argument("--stageA_kd_temp", type=float, default=2.5)
    ap.add_argument("--stageA_lambda_kd", type=float, default=1.0)
    ap.add_argument("--stageA_lambda_emb", type=float, default=1.2)
    ap.add_argument("--stageA_lambda_tok", type=float, default=0.6)
    ap.add_argument("--stageA_lambda_phys", type=float, default=0.2)
    ap.add_argument("--stageA_lambda_budget_hinge", type=float, default=0.03)
    ap.add_argument("--stageA_budget_eps", type=float, default=0.015)
    ap.add_argument("--stageA_budget_weight_floor", type=float, default=1e-4)
    ap.add_argument("--stageA_target_tpr", type=float, default=0.50)
    ap.add_argument("--disable_stageA_loss_normalization", action="store_true")
    ap.add_argument("--stageA_loss_norm_ema_decay", type=float, default=0.98)
    ap.add_argument("--stageA_loss_norm_eps", type=float, default=1e-6)
    ap.add_argument("--disable_stageA_stagewise_best_reload", action="store_true")
    ap.add_argument("--stageA_lambda_delta", type=float, default=0.00)
    ap.add_argument("--stageA_delta_tau", type=float, default=0.05)
    ap.add_argument("--stageA_delta_lambda_fp", type=float, default=3.0)

    ap.add_argument("--added_target_scale", type=float, default=0.90)
    ap.add_argument("--reco_weight_threshold", type=float, default=0.03)
    ap.add_argument("--reco_eval_batch_size", type=int, default=256)

    # Concat joint (replaces corrected-only joint + dual in this variant)
    ap.add_argument("--stageC_epochs", type=int, default=65)
    ap.add_argument("--stageC_patience", type=int, default=14)
    ap.add_argument("--stageC_min_epochs", type=int, default=25)
    ap.add_argument("--stageC_lr_model", type=float, default=2e-4)
    ap.add_argument("--stageC_lr_reco", type=float, default=1e-4)
    ap.add_argument("--stageC_weight_decay", type=float, default=1e-4)
    ap.add_argument("--stageC_warmup_epochs", type=int, default=3)
    ap.add_argument("--stageC_lambda_reco", type=float, default=0.4)
    ap.add_argument("--stageC_lambda_rank", type=float, default=0.0)
    ap.add_argument("--stageC_lambda_cons", type=float, default=0.06)
    ap.add_argument("--joint_select_metric", type=str, default="auc", choices=["auc", "fpr50"])

    ap.add_argument("--report_target_tpr", type=float, default=0.50)
    ap.add_argument("--combo_weight_step", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--use_corrected_flags", action="store_true")
    ap.add_argument("--save_fusion_scores", action="store_true")
    args = ap.parse_args()

    b.set_seed(int(args.seed))

    cfg = _deepcopy_cfg()
    cfg["hlt_effects"]["merge_radius"] = float(args.merge_radius)
    cfg["hlt_effects"]["eff_plateau_barrel"] = float(args.eff_plateau_barrel)
    cfg["hlt_effects"]["eff_plateau_endcap"] = float(args.eff_plateau_endcap)
    cfg["hlt_effects"]["smear_a"] = float(args.smear_a)
    cfg["hlt_effects"]["smear_b"] = float(args.smear_b)
    cfg["hlt_effects"]["smear_c"] = float(args.smear_c)

    cfg["reconstructor_training"]["epochs"] = int(args.stageA_epochs)
    cfg["reconstructor_training"]["patience"] = int(args.stageA_patience)
    phase035 = int(args.stageA_phase035_epochs)
    phase070 = int(args.stageA_phase070_epochs)
    if phase035 == 0 and phase070 == 0:
        cfg["reconstructor_training"]["stage1_epochs"] = 0
        cfg["reconstructor_training"]["stage2_epochs"] = 0
    elif phase035 > 0 or phase070 > 0:
        if phase035 <= 0 or phase070 <= 0:
            raise ValueError(
                "When using custom Stage-A curriculum phase lengths, either set both "
                "--stageA_phase035_epochs and --stageA_phase070_epochs to 0 "
                "(to skip warmup phases), or set both to > 0."
            )
        stage1 = phase035
        stage2 = phase035 + phase070
        if stage2 >= int(args.stageA_epochs):
            raise ValueError(
                f"Custom Stage-A phase lengths must leave room for phase_100: "
                f"phase035+phase070={stage2} must be < stageA_epochs={int(args.stageA_epochs)}"
            )
        cfg["reconstructor_training"]["stage1_epochs"] = int(stage1)
        cfg["reconstructor_training"]["stage2_epochs"] = int(stage2)

    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(list(train_path.glob("*.h5")))
    else:
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = int(args.offset_jets) + int(args.n_train_jets)
    print("Loading offline constituents...")
    all_const_full, all_labels_full = b.load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[args.offset_jets: args.offset_jets + args.n_train_jets]
    labels = all_labels_full[args.offset_jets: args.offset_jets + args.n_train_jets].astype(np.int64)

    raw_mask = const_raw[:, :, 0] > 0.0
    masks_off = raw_mask & (const_raw[:, :, 0] >= float(cfg["hlt_effects"]["pt_threshold_offline"]))
    const_off = const_raw.copy()
    const_off[~masks_off] = 0.0

    print("Generating pseudo-HLT...")
    hlt_const, hlt_mask, hlt_stats, _budget_truth = b.apply_hlt_effects_realistic_nomap(
        const_off, masks_off, cfg, seed=int(args.seed)
    )

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    rho = b._clamp_target_scale(float(args.added_target_scale))
    budget_merge_true = (rho * true_added_raw).astype(np.float32)
    budget_eff_true = ((1.0 - rho) * true_added_raw).astype(np.float32)

    print(
        f"Non-priv rho split setup: rho={rho:.3f}, "
        f"mean_true_added_raw={float(true_added_raw.mean()):.3f}, "
        f"mean_target_merge={float(budget_merge_true.mean()):.3f}, "
        f"mean_target_eff={float(budget_eff_true.mean()):.3f}"
    )

    print("Computing features...")
    feat_off = b.compute_features(const_off, masks_off)
    feat_hlt = b.compute_features(hlt_const, hlt_mask)

    max_concat_constits = int(args.max_concat_constits)
    const_concat, mask_concat = m6.build_concat_constituents(
        const_off=const_off,
        mask_off=masks_off,
        const_hlt=hlt_const,
        mask_hlt=hlt_mask,
        max_concat_constits=max_concat_constits,
    )
    feat_concat = b.compute_features(const_concat, mask_concat)

    idx = np.arange(len(labels))
    total_need = int(args.n_train_split + args.n_val_split + args.n_test_split)
    if total_need > len(idx):
        raise ValueError(f"Requested split counts exceed available jets: {total_need} > {len(idx)}")
    if total_need < len(idx):
        idx_use, _ = train_test_split(
            idx, train_size=total_need, random_state=int(args.seed), stratify=labels[idx]
        )
    else:
        idx_use = idx

    train_idx, rem_idx = train_test_split(
        idx_use,
        train_size=int(args.n_train_split),
        random_state=int(args.seed),
        stratify=labels[idx_use],
    )
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=int(args.n_val_split),
        test_size=int(args.n_test_split),
        random_state=int(args.seed),
        stratify=labels[rem_idx],
    )
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} (custom_counts=True)")

    means_off, stds_off = b.get_stats(feat_off, masks_off, train_idx)
    feat_off_std = b.standardize(feat_off, masks_off, means_off, stds_off)
    feat_hlt_std = b.standardize(feat_hlt, hlt_mask, means_off, stds_off)
    means_concat, stds_concat = b.get_stats(feat_concat, mask_concat, train_idx)
    feat_concat_std = b.standardize(feat_concat, mask_concat, means_concat, stds_concat)

    data_setup = {
        "train_path_arg": str(args.train_path),
        "train_files": [str(p.resolve()) for p in train_files],
        "n_train_jets": int(args.n_train_jets),
        "offset_jets": int(args.offset_jets),
        "max_constits": int(args.max_constits),
        "max_concat_constits": int(max_concat_constits),
        "seed": int(args.seed),
        "split": {
            "mode": "custom_counts",
            "n_train_split": int(len(train_idx)),
            "n_val_split": int(len(val_idx)),
            "n_test_split": int(len(test_idx)),
        },
        "hlt_effects": cfg["hlt_effects"],
        "variant": "concat_teacher_hltreco_stagea_then_concatjoint",
        "stageA_phase035_epochs": int(args.stageA_phase035_epochs),
        "stageA_phase070_epochs": int(args.stageA_phase070_epochs),
        "rho": float(rho),
        "mean_true_added_raw": float(true_added_raw.mean()),
        "mean_target_merge": float(budget_merge_true.mean()),
        "mean_target_eff": float(budget_eff_true.mean()),
    }
    with open(save_root / "data_setup.json", "w", encoding="utf-8") as f:
        json.dump(data_setup, f, indent=2)
    np.savez_compressed(
        save_root / "data_splits.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        means_off=means_off.astype(np.float32),
        stds_off=stds_off.astype(np.float32),
        means_concat=means_concat.astype(np.float32),
        stds_concat=stds_concat.astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("STEP 1: HLT BASELINE + CONCAT TEACHER")
    print("=" * 70)
    bs_cls = int(args.batch_size)
    ds_train_hlt = b.JetDataset(feat_hlt_std[train_idx], hlt_mask[train_idx], labels[train_idx])
    ds_val_hlt = b.JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = b.JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_train_hlt = DataLoader(ds_train_hlt, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=bs_cls, shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=bs_cls, shuffle=False)

    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline = b.train_single_view_classifier_auc(
        baseline, dl_train_hlt, dl_val_hlt, device, cfg["training"], name="Baseline-HLT"
    )
    auc_hlt_test, preds_hlt_test, labs_hlt_test = b.eval_classifier(baseline, dl_test_hlt, device)
    auc_hlt_val, preds_hlt_val, labs_hlt_val = b.eval_classifier(baseline, dl_val_hlt, device)

    ds_train_concat = b.JetDataset(feat_concat_std[train_idx], mask_concat[train_idx], labels[train_idx])
    ds_val_concat = b.JetDataset(feat_concat_std[val_idx], mask_concat[val_idx], labels[val_idx])
    ds_test_concat = b.JetDataset(feat_concat_std[test_idx], mask_concat[test_idx], labels[test_idx])
    dl_train_concat = DataLoader(ds_train_concat, batch_size=bs_cls, shuffle=True, drop_last=True)
    dl_val_concat = DataLoader(ds_val_concat, batch_size=bs_cls, shuffle=False)
    dl_test_concat = DataLoader(ds_test_concat, batch_size=bs_cls, shuffle=False)

    concat_teacher = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    concat_teacher = b.train_single_view_classifier_auc(
        concat_teacher, dl_train_concat, dl_val_concat, device, cfg["training"], name="ConcatTeacher"
    )
    auc_concat_test, preds_concat_test, labs_concat_test = b.eval_classifier(concat_teacher, dl_test_concat, device)
    auc_concat_val, preds_concat_val, labs_concat_val = b.eval_classifier(concat_teacher, dl_val_concat, device)

    assert np.array_equal(labs_hlt_val.astype(np.float32), labs_concat_val.astype(np.float32))
    assert np.array_equal(labs_hlt_test.astype(np.float32), labs_concat_test.astype(np.float32))

    hlt_thr_prob, hlt_thr_tpr, hlt_thr_fpr = m6.threshold_at_target_tpr(
        labs_hlt_val.astype(np.float32), preds_hlt_val.astype(np.float64), float(args.stageA_target_tpr)
    )
    print(
        f"StageA delta HLT reference @TPR={float(args.stageA_target_tpr):.2f}: "
        f"threshold_prob={hlt_thr_prob:.6f}, val_tpr={hlt_thr_tpr:.6f}, val_fpr={hlt_thr_fpr:.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 2: STAGE A (CONCAT-TEACHER-GUIDED RECONSTRUCTOR PRETRAIN)")
    print("=" * 70)
    ds_train_reco = m6.StageAConcatTeacherDataset(
        feat_hlt=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        const_teacher=const_concat[train_idx],
        mask_teacher=mask_concat[train_idx],
        labels=labels[train_idx],
        budget_merge_true=budget_merge_true[train_idx],
        budget_eff_true=budget_eff_true[train_idx],
    )
    ds_val_reco = m6.StageAConcatTeacherDataset(
        feat_hlt=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        const_teacher=const_concat[val_idx],
        mask_teacher=mask_concat[val_idx],
        labels=labels[val_idx],
        budget_merge_true=budget_merge_true[val_idx],
        budget_eff_true=budget_eff_true[val_idx],
    )
    dl_train_reco = DataLoader(
        ds_train_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val_reco = DataLoader(
        ds_val_reco,
        batch_size=int(cfg["reconstructor_training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Monkey-patch m6 Stage-A loss to the requested concat(HLT+reco_pred) formulation.
    m6._compute_concat_teacher_guided_reco_losses = _compute_concat_teacher_guided_reco_losses_hltreco

    reconstructor = b.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    b.BASE_CONFIG["loss"] = cfg["loss"]
    reconstructor, reco_val_metrics = m6.train_reconstructor_concat_teacher_stagewise(
        model=reconstructor,
        train_loader=dl_train_reco,
        val_loader=dl_val_reco,
        device=device,
        train_cfg=cfg["reconstructor_training"],
        loss_cfg=cfg["loss"],
        teacher_model=concat_teacher,
        hlt_model=baseline,
        hlt_threshold_prob=float(hlt_thr_prob),
        feat_means=means_concat.astype(np.float32),
        feat_stds=stds_concat.astype(np.float32),
        kd_temperature=float(args.stageA_kd_temp),
        lambda_kd=float(args.stageA_lambda_kd),
        lambda_emb=float(args.stageA_lambda_emb),
        lambda_tok=float(args.stageA_lambda_tok),
        lambda_phys=float(args.stageA_lambda_phys),
        lambda_budget_hinge=float(args.stageA_lambda_budget_hinge),
        lambda_delta=float(args.stageA_lambda_delta),
        delta_tau=float(args.stageA_delta_tau),
        delta_lambda_fp=float(args.stageA_delta_lambda_fp),
        budget_eps=float(args.stageA_budget_eps),
        budget_weight_floor=float(args.stageA_budget_weight_floor),
        target_tpr_for_fpr=float(args.stageA_target_tpr),
        normalize_loss_terms=not bool(args.disable_stageA_loss_normalization),
        loss_norm_ema_decay=float(args.stageA_loss_norm_ema_decay),
        loss_norm_eps=float(args.stageA_loss_norm_eps),
        reload_best_at_stage_transition=not bool(args.disable_stageA_stagewise_best_reload),
    )

    print("\n" + "=" * 70)
    print("STEP 3: CONCAT-TEACHER ON CONCAT(HLT + RECO) VIEW")
    print("=" * 70)
    auc_reco_teacher_val, preds_reco_teacher_val, labs_reco_val, fpr50_reco_teacher_val = eval_teacher_on_concat_hltreco_split(
        reconstructor=reconstructor,
        teacher=concat_teacher,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        labels=labels,
        split_idx=val_idx,
        means=means_concat,
        stds=stds_concat,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        weight_floor=float(args.reco_weight_threshold),
        max_concat_constits=int(max_concat_constits),
        target_tpr=float(args.report_target_tpr),
    )
    auc_reco_teacher_test, preds_reco_teacher_test, labs_reco_test, fpr50_reco_teacher_test = eval_teacher_on_concat_hltreco_split(
        reconstructor=reconstructor,
        teacher=concat_teacher,
        feat_hlt_std=feat_hlt_std,
        hlt_mask=hlt_mask,
        hlt_const=hlt_const,
        labels=labels,
        split_idx=test_idx,
        means=means_concat,
        stds=stds_concat,
        device=device,
        batch_size=int(args.reco_eval_batch_size),
        weight_floor=float(args.reco_weight_threshold),
        max_concat_constits=int(max_concat_constits),
        target_tpr=float(args.report_target_tpr),
    )

    print("\n" + "=" * 70)
    print("STEP 4: CONCAT(HLT + RECO) TAGGER (FROZEN STAGE-A RECO)")
    print("=" * 70)
    corrected_use_flags = bool(args.use_corrected_flags)
    feat_corr_all, mask_corr_all = b.build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=int(args.batch_size),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )
    feat_concat_hltreco_all, mask_concat_hltreco_all = _build_concat_view_numpy(
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        feat_corr=feat_corr_all,
        mask_corr=mask_corr_all,
    )

    ds_train_concat_hltreco = b.JetDataset(
        feat_concat_hltreco_all[train_idx], mask_concat_hltreco_all[train_idx], labels[train_idx]
    )
    ds_val_concat_hltreco = b.JetDataset(
        feat_concat_hltreco_all[val_idx], mask_concat_hltreco_all[val_idx], labels[val_idx]
    )
    ds_test_concat_hltreco = b.JetDataset(
        feat_concat_hltreco_all[test_idx], mask_concat_hltreco_all[test_idx], labels[test_idx]
    )
    dl_train_concat_hltreco = DataLoader(ds_train_concat_hltreco, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    dl_val_concat_hltreco = DataLoader(ds_val_concat_hltreco, batch_size=int(args.batch_size), shuffle=False)
    dl_test_concat_hltreco = DataLoader(ds_test_concat_hltreco, batch_size=int(args.batch_size), shuffle=False)

    concat_hltreco_model = b.ParticleTransformer(input_dim=int(feat_concat_hltreco_all.shape[-1]), **cfg["model"]).to(device)
    concat_hltreco_model = b.train_single_view_classifier_auc(
        concat_hltreco_model,
        dl_train_concat_hltreco,
        dl_val_concat_hltreco,
        device,
        cfg["training"],
        name="ConcatHLTReco-PostStageA",
    )
    auc_corr_val, preds_corr_val, labs_corr_val = b.eval_classifier(concat_hltreco_model, dl_val_concat_hltreco, device)
    auc_corr_test, preds_corr_test, labs_corr_test = b.eval_classifier(concat_hltreco_model, dl_test_concat_hltreco, device)

    print("\n" + "=" * 70)
    print("STEP 5: CONCAT(HLT + RECO) JOINT FINETUNE (PRE->POST)")
    print("=" * 70)
    ds_train_joint = ConcatJointDataset(
        feat_hlt_reco=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        budget_merge_true=budget_merge_true[train_idx],
        budget_eff_true=budget_eff_true[train_idx],
        labels=labels[train_idx],
    )
    ds_val_joint = ConcatJointDataset(
        feat_hlt_reco=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        budget_merge_true=budget_merge_true[val_idx],
        budget_eff_true=budget_eff_true[val_idx],
        labels=labels[val_idx],
    )
    ds_test_joint = ConcatJointDataset(
        feat_hlt_reco=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        const_off=const_off[test_idx],
        mask_off=masks_off[test_idx],
        budget_merge_true=budget_merge_true[test_idx],
        budget_eff_true=budget_eff_true[test_idx],
        labels=labels[test_idx],
    )
    dl_train_joint = DataLoader(
        ds_train_joint, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=args.num_workers
    )
    dl_val_joint = DataLoader(ds_val_joint, batch_size=int(args.batch_size), shuffle=False, num_workers=args.num_workers)
    dl_test_joint = DataLoader(ds_test_joint, batch_size=int(args.batch_size), shuffle=False, num_workers=args.num_workers)

    reconstructor, concat_hltreco_model, joint_metrics, joint_states = train_concat_joint(
        reconstructor=reconstructor,
        concat_model=concat_hltreco_model,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageC-ConcatHLTRecoJoint",
        epochs=int(args.stageC_epochs),
        patience=int(args.stageC_patience),
        min_epochs=int(args.stageC_min_epochs),
        lr_model=float(args.stageC_lr_model),
        lr_reco=float(args.stageC_lr_reco),
        weight_decay=float(args.stageC_weight_decay),
        warmup_epochs=int(args.stageC_warmup_epochs),
        lambda_reco=float(args.stageC_lambda_reco),
        lambda_rank=float(args.stageC_lambda_rank),
        lambda_cons=float(args.stageC_lambda_cons),
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
        select_metric=str(args.joint_select_metric),
    )

    auc_joint_val, preds_joint_val, labs_joint_val, fpr50_joint_val = eval_concat_joint_model(
        reconstructor=reconstructor,
        concat_model=concat_hltreco_model,
        loader=dl_val_joint,
        device=device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )
    auc_joint_test, preds_joint_test, labs_joint_test, fpr50_joint_test = eval_concat_joint_model(
        reconstructor=reconstructor,
        concat_model=concat_hltreco_model,
        loader=dl_test_joint,
        device=device,
        corrected_weight_floor=float(args.reco_weight_threshold),
        corrected_use_flags=corrected_use_flags,
    )

    fpr_hlt, tpr_hlt, _ = roc_curve(labs_hlt_test, preds_hlt_test)
    fpr_corr, tpr_corr, _ = roc_curve(labs_corr_test, preds_corr_test)
    fpr_joint, tpr_joint, _ = roc_curve(labs_joint_test, preds_joint_test)
    fpr50_hlt = float(b.fpr_at_target_tpr(fpr_hlt, tpr_hlt, 0.50))
    fpr50_corr = float(b.fpr_at_target_tpr(fpr_corr, tpr_corr, 0.50))
    fpr50_joint = float(b.fpr_at_target_tpr(fpr_joint, tpr_joint, 0.50))

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print(f"HLT baseline AUC (val/test): {auc_hlt_val:.4f} / {auc_hlt_test:.4f}")
    print(f"ConcatTeacher AUC (val/test): {auc_concat_val:.4f} / {auc_concat_test:.4f}")
    print(f"RecoTeacherConcatSoft AUC (val/test): {auc_reco_teacher_val:.4f} / {auc_reco_teacher_test:.4f}")
    print(f"ConcatHLTReco pre-joint AUC (val/test): {auc_corr_val:.4f} / {auc_corr_test:.4f}")
    print(f"ConcatHLTReco post-joint AUC (val/test): {auc_joint_val:.4f} / {auc_joint_test:.4f}")
    print(
        "FPR@50 HLT / ConcatTeacher / RecoTeacherConcatSoft / ConcatHLTReco(pre) / ConcatHLTReco(post): "
        f"{fpr50_hlt:.6f} / {float(b.fpr_at_target_tpr(*roc_curve(labs_concat_test, preds_concat_test)[:2], 0.50)):.6f} / "
        f"{fpr50_reco_teacher_test:.6f} / {fpr50_corr:.6f} / {fpr50_joint:.6f}"
    )

    # Combo reports (same style as m6)
    combo_corr_valsel, combo_corr_oracle = m6._combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_other_val=preds_corr_val.astype(np.float64),
        preds_other_test=preds_corr_test.astype(np.float64),
        other_name="concat_hltreco_pre",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )
    combo_joint_valsel, combo_joint_oracle = m6._combo_reports(
        labels_val=labs_hlt_val.astype(np.float32),
        labels_test=labs_hlt_test.astype(np.float32),
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_other_val=preds_joint_val.astype(np.float64),
        preds_other_test=preds_joint_test.astype(np.float64),
        other_name="concat_hltreco_post",
        target_tpr=float(args.report_target_tpr),
        weight_step=float(args.combo_weight_step),
    )

    np.savez_compressed(
        save_root / "concat_hltreco_stagea_then_concatjoint_scores.npz",
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_concat_teacher_val=preds_concat_val.astype(np.float64),
        preds_concat_teacher_test=preds_concat_test.astype(np.float64),
        preds_reco_teacher_val=preds_reco_teacher_val.astype(np.float64),
        preds_reco_teacher_test=preds_reco_teacher_test.astype(np.float64),
        preds_concat_hltreco_pre_val=preds_corr_val.astype(np.float64),
        preds_concat_hltreco_pre_test=preds_corr_test.astype(np.float64),
        preds_concat_hltreco_post_val=preds_joint_val.astype(np.float64),
        preds_concat_hltreco_post_test=preds_joint_test.astype(np.float64),
        labs_val=labs_hlt_val.astype(np.float32),
        labs_test=labs_hlt_test.astype(np.float32),
    )

    report = {
        "variant": "concat_teacher_hltreco_stagea_then_concatjoint",
        "settings": {
            "seed": int(args.seed),
            "max_constits": int(args.max_constits),
            "max_concat_constits": int(max_concat_constits),
            "stageA_lambda_kd": float(args.stageA_lambda_kd),
            "stageA_lambda_emb": float(args.stageA_lambda_emb),
            "stageA_lambda_tok": float(args.stageA_lambda_tok),
            "stageA_lambda_phys": float(args.stageA_lambda_phys),
            "stageA_lambda_budget_hinge": float(args.stageA_lambda_budget_hinge),
            "stageC_lambda_reco": float(args.stageC_lambda_reco),
            "stageC_lambda_rank": float(args.stageC_lambda_rank),
            "stageC_lambda_cons": float(args.stageC_lambda_cons),
            "joint_select_metric": str(args.joint_select_metric),
        },
        "metrics": {
            "hlt": {"auc_val": float(auc_hlt_val), "auc_test": float(auc_hlt_test), "fpr50_test": float(fpr50_hlt)},
            "concat_teacher": {
                "auc_val": float(auc_concat_val),
                "auc_test": float(auc_concat_test),
                "fpr50_test": float(b.fpr_at_target_tpr(*roc_curve(labs_concat_test, preds_concat_test)[:2], 0.50)),
            },
            "reco_teacher_concatsoft": {
                "auc_val": float(auc_reco_teacher_val),
                "auc_test": float(auc_reco_teacher_test),
                "fpr50_val": float(fpr50_reco_teacher_val),
                "fpr50_test": float(fpr50_reco_teacher_test),
            },
            "concat_hltreco_pre_joint": {
                "auc_val": float(auc_corr_val),
                "auc_test": float(auc_corr_test),
                "fpr50_test": float(fpr50_corr),
            },
            "concat_hltreco_post_joint": {
                "auc_val": float(auc_joint_val),
                "auc_test": float(auc_joint_test),
                "fpr50_test": float(fpr50_joint),
                "joint_selection": joint_metrics,
            },
            "combo": {
                "pre_val_selected": combo_corr_valsel,
                "pre_test_oracle": combo_corr_oracle,
                "post_val_selected": combo_joint_valsel,
                "post_test_oracle": combo_joint_oracle,
            },
        },
    }
    with open(save_root / "concat_hltreco_stagea_then_concatjoint_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not bool(args.skip_save_models):
        torch.save({"model": baseline.state_dict(), "auc": float(auc_hlt_test)}, save_root / "baseline.pt")
        torch.save({"model": concat_teacher.state_dict(), "auc": float(auc_concat_test)}, save_root / "concat_teacher.pt")
        torch.save({"model": reconstructor.state_dict(), "val": reco_val_metrics}, save_root / "offline_reconstructor_stageA.pt")
        torch.save({"model": concat_hltreco_model.state_dict(), "val": joint_metrics}, save_root / "concat_hltreco_joint.pt")
        if joint_states.get("fpr50", {}).get("model") is not None:
            torch.save(
                {"model": joint_states["fpr50"]["model"], "val": joint_metrics},
                save_root / "concat_hltreco_joint_bestfpr50.pt",
            )
            torch.save(
                {"model": joint_states["fpr50"]["reco"], "val": joint_metrics},
                save_root / "offline_reconstructor_stageA_bestfpr50.pt",
            )

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_val_test.npz",
            preds_hlt_val=preds_hlt_val.astype(np.float64),
            preds_hlt_test=preds_hlt_test.astype(np.float64),
            preds_joint_val=preds_joint_val.astype(np.float64),
            preds_joint_test=preds_joint_test.astype(np.float64),
            labels_val=labs_hlt_val.astype(np.float32),
            labels_test=labs_hlt_test.astype(np.float32),
        )
        print(f"Saved fusion score arrays to: {save_root / 'fusion_scores_val_test.npz'}")

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()
