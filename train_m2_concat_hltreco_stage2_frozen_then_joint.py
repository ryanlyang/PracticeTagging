#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from unmerge_correct_hlt import (
    compute_features,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    apply_hlt_effects_realistic_nomap,
)
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as b
from analyze_m2_router_signal_sweep import _build_train_file_list, _load_ckpt_state


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _offline_mask(const_raw: np.ndarray, pt_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(pt_thr))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0
    return const_off.astype(np.float32), mask_off.astype(bool)


def _fpr_at_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(b.fpr_at_target_tpr(fpr, tpr, float(target_tpr)))


def _metrics_from_scores(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan")
    return {
        "auc": auc,
        "fpr30": _fpr_at_tpr(y, p, 0.30),
        "fpr50": _fpr_at_tpr(y, p, 0.50),
    }


def _build_concat_view_numpy(
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    feat_corr: np.ndarray,
    mask_corr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # HLT: [B,L,7], corrected: [B,Lc,Dc] (typically Dc=10 or 12)
    if feat_hlt.ndim != 3 or feat_corr.ndim != 3:
        raise ValueError("Expected rank-3 features for hlt/corrected")
    if feat_hlt.shape[0] != feat_corr.shape[0]:
        raise ValueError("Batch mismatch in concat view builder")
    if mask_hlt.shape[:2] != feat_hlt.shape[:2]:
        raise ValueError("mask_hlt shape mismatch")
    if mask_corr.shape[:2] != feat_corr.shape[:2]:
        raise ValueError("mask_corr shape mismatch")

    bsz, lh, dh = feat_hlt.shape
    _, lc, dc = feat_corr.shape
    d_base = max(int(dh), int(dc))

    h_pad = np.zeros((bsz, lh, d_base), dtype=np.float32)
    c_pad = np.zeros((bsz, lc, d_base), dtype=np.float32)
    h_pad[:, :, :dh] = feat_hlt.astype(np.float32)
    c_pad[:, :, :dc] = feat_corr.astype(np.float32)

    # Append source flag (0=HLT, 1=Reco-derived).
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
    freeze_reconstructor: bool,
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
        p.requires_grad = not freeze_reconstructor

    params = [{"params": concat_model.parameters(), "lr": float(lr_model)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

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
        if freeze_reconstructor:
            reconstructor.eval()
        else:
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

            if freeze_reconstructor:
                with torch.no_grad():
                    reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
            else:
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
                reco_losses = b.compute_reconstruction_losses_weighted(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    b.BASE_CONFIG["loss"],
                    sample_weight=None,
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
            if not freeze_reconstructor:
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

        print_every = 1 if str(stage_name).startswith("StageC") else 5
        if (ep + 1) % print_every == 0:
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
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--save_dir", type=str, default="checkpoints/reco_teacher_joint_fusion_6model_150k75k150k/model2_concat_hltreco_stage2")
    ap.add_argument("--run_name", type=str, default="model2_concat_hltreco_stage2_150k75k150k_seed0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--skip_save_models", action="store_true")

    ap.add_argument("--reco_ckpt", type=str, default="")
    ap.add_argument("--baseline_ckpt", type=str, default="")
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--use_corrected_flags", action="store_true")

    ap.add_argument("--pretrain_epochs", type=int, default=45)
    ap.add_argument("--pretrain_patience", type=int, default=10)
    ap.add_argument("--joint_epochs", type=int, default=25)
    ap.add_argument("--joint_patience", type=int, default=8)
    ap.add_argument("--joint_min_epochs", type=int, default=8)
    ap.add_argument("--joint_lr_model", type=float, default=2e-4)
    ap.add_argument("--joint_lr_reco", type=float, default=1e-4)
    ap.add_argument("--joint_weight_decay", type=float, default=1e-4)
    ap.add_argument("--joint_warmup_epochs", type=int, default=3)
    ap.add_argument("--joint_lambda_reco", type=float, default=0.4)
    ap.add_argument("--joint_lambda_rank", type=float, default=0.0)
    ap.add_argument("--joint_lambda_cons", type=float, default=0.06)
    ap.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])

    ap.add_argument("--save_fusion_scores", action="store_true")
    args = ap.parse_args()

    b.set_seed(int(args.seed))
    run_dir = Path(args.run_dir).expanduser().resolve()
    save_root = Path(args.save_dir).expanduser().resolve() / str(args.run_name)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Save dir: {save_root}")

    setup_path = run_dir / "data_setup.json"
    splits_path = run_dir / "data_splits.npz"
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing {setup_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}")

    with setup_path.open("r", encoding="utf-8") as f:
        setup = json.load(f)
    splits = np.load(splits_path, allow_pickle=False)
    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)
    test_idx = splits["test_idx"].astype(np.int64)
    means = splits["means"].astype(np.float32) if "means" in splits.files else splits["means_off"].astype(np.float32)
    stds = splits["stds"].astype(np.float32) if "stds" in splits.files else splits["stds_off"].astype(np.float32)

    cfg = _deepcopy_cfg()
    if "hlt_effects" in setup:
        cfg["hlt_effects"].update(setup["hlt_effects"])

    n_train_jets = int(setup["n_train_jets"])
    offset_jets = int(setup["offset_jets"])
    max_constits = int(args.max_constits)
    hlt_seed = int(setup.get("seed", 0))

    train_files: List[Path] = _build_train_file_list(setup, args.train_path)
    max_jets_needed = int(offset_jets + n_train_jets)

    print("Loading offline constituents...")
    all_const_full, all_labels_full = load_raw_constituents_from_h5(
        train_files,
        max_jets=max_jets_needed,
        max_constits=max_constits,
    )
    if all_const_full.shape[0] < max_jets_needed:
        raise RuntimeError(f"Not enough jets: requested {max_jets_needed}, got {all_const_full.shape[0]}")

    const_raw = all_const_full[offset_jets : offset_jets + n_train_jets]
    labels = all_labels_full[offset_jets : offset_jets + n_train_jets].astype(np.int64)

    print("Generating pseudo-HLT...")
    const_off, masks_off = _offline_mask(const_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    hlt_const, hlt_mask, _hlt_stats, _budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=hlt_seed,
    )

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)
    rho = float(setup.get("added_target_scale", setup.get("rho", 0.90)))
    rho = float(np.clip(rho, 0.0, 1.0))
    budget_merge_true = (rho * true_added_raw).astype(np.float32)
    budget_eff_true = ((1.0 - rho) * true_added_raw).astype(np.float32)

    print("Computing features...")
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds).astype(np.float32)

    reco_ckpt = Path(args.reco_ckpt).expanduser().resolve() if str(args.reco_ckpt).strip() else (run_dir / "offline_reconstructor_stage2.pt")
    if not reco_ckpt.exists():
        raise FileNotFoundError(f"Missing reconstructor checkpoint: {reco_ckpt}")
    baseline_ckpt = Path(args.baseline_ckpt).expanduser().resolve() if str(args.baseline_ckpt).strip() else (run_dir / "baseline.pt")
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {baseline_ckpt}")

    reconstructor = b.OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(_load_ckpt_state(reco_ckpt, device), strict=True)
    reconstructor.eval()

    baseline = b.ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline.load_state_dict(_load_ckpt_state(baseline_ckpt, device), strict=True)
    baseline.eval()

    print("\n" + "=" * 70)
    print("STEP 1: BASELINE HLT EVAL")
    print("=" * 70)
    ds_val_hlt = b.JetDataset(feat_hlt_std[val_idx], hlt_mask[val_idx], labels[val_idx])
    ds_test_hlt = b.JetDataset(feat_hlt_std[test_idx], hlt_mask[test_idx], labels[test_idx])
    dl_val_hlt = DataLoader(ds_val_hlt, batch_size=int(args.batch_size), shuffle=False)
    dl_test_hlt = DataLoader(ds_test_hlt, batch_size=int(args.batch_size), shuffle=False)
    auc_hlt_val, preds_hlt_val, labs_hlt_val = b.eval_classifier(baseline, dl_val_hlt, device)
    auc_hlt_test, preds_hlt_test, labs_hlt_test = b.eval_classifier(baseline, dl_test_hlt, device)
    m_hlt_val = _metrics_from_scores(labs_hlt_val, preds_hlt_val)
    m_hlt_test = _metrics_from_scores(labs_hlt_test, preds_hlt_test)

    print("\n" + "=" * 70)
    print("STEP 2: PRE-JOINT CONCAT TAGGER (FROZEN STAGE2 RECO)")
    print("=" * 70)
    feat_corr_all, mask_corr_all = b.build_corrected_view_numpy(
        reconstructor=reconstructor,
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        device=device,
        batch_size=int(args.batch_size),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    feat_concat_all, mask_concat_all = _build_concat_view_numpy(
        feat_hlt=feat_hlt_std,
        mask_hlt=hlt_mask,
        feat_corr=feat_corr_all,
        mask_corr=mask_corr_all,
    )

    ds_train_concat = b.JetDataset(feat_concat_all[train_idx], mask_concat_all[train_idx], labels[train_idx])
    ds_val_concat = b.JetDataset(feat_concat_all[val_idx], mask_concat_all[val_idx], labels[val_idx])
    ds_test_concat = b.JetDataset(feat_concat_all[test_idx], mask_concat_all[test_idx], labels[test_idx])
    dl_train_concat = DataLoader(ds_train_concat, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    dl_val_concat = DataLoader(ds_val_concat, batch_size=int(args.batch_size), shuffle=False)
    dl_test_concat = DataLoader(ds_test_concat, batch_size=int(args.batch_size), shuffle=False)

    cfg_cls = json.loads(json.dumps(cfg["training"]))
    cfg_cls["epochs"] = int(args.pretrain_epochs)
    cfg_cls["patience"] = int(args.pretrain_patience)

    concat_model = b.ParticleTransformer(input_dim=int(feat_concat_all.shape[-1]), **cfg["model"]).to(device)
    concat_model = b.train_single_view_classifier_auc(
        concat_model,
        dl_train_concat,
        dl_val_concat,
        device,
        cfg_cls,
        name="ConcatHLTReco-PreJoint",
    )
    auc_concat_pre_val, preds_concat_pre_val, labs_concat_pre_val = b.eval_classifier(concat_model, dl_val_concat, device)
    auc_concat_pre_test, preds_concat_pre_test, labs_concat_pre_test = b.eval_classifier(concat_model, dl_test_concat, device)
    m_pre_val = _metrics_from_scores(labs_concat_pre_val, preds_concat_pre_val)
    m_pre_test = _metrics_from_scores(labs_concat_pre_test, preds_concat_pre_test)

    print("\n" + "=" * 70)
    print("STEP 3: JOINT FINETUNE CONCAT (UNFREEZE RECO + CONCAT TAGGER)")
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
    dl_train_joint = DataLoader(ds_train_joint, batch_size=int(args.batch_size), shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    dl_val_joint = DataLoader(ds_val_joint, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_test_joint = DataLoader(ds_test_joint, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    reconstructor, concat_model, joint_metrics, joint_states = train_concat_joint(
        reconstructor=reconstructor,
        concat_model=concat_model,
        train_loader=dl_train_joint,
        val_loader=dl_val_joint,
        device=device,
        stage_name="StageC-ConcatJoint",
        freeze_reconstructor=False,
        epochs=int(args.joint_epochs),
        patience=int(args.joint_patience),
        min_epochs=int(args.joint_min_epochs),
        lr_model=float(args.joint_lr_model),
        lr_reco=float(args.joint_lr_reco),
        weight_decay=float(args.joint_weight_decay),
        warmup_epochs=int(args.joint_warmup_epochs),
        lambda_reco=float(args.joint_lambda_reco),
        lambda_rank=float(args.joint_lambda_rank),
        lambda_cons=float(args.joint_lambda_cons),
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
        select_metric=str(args.selection_metric),
    )

    auc_concat_joint_val, preds_concat_joint_val, labs_concat_joint_val, fpr50_concat_joint_val = eval_concat_joint_model(
        reconstructor, concat_model, dl_val_joint, device, float(args.corrected_weight_floor), bool(args.use_corrected_flags)
    )
    auc_concat_joint_test, preds_concat_joint_test, labs_concat_joint_test, fpr50_concat_joint_test = eval_concat_joint_model(
        reconstructor, concat_model, dl_test_joint, device, float(args.corrected_weight_floor), bool(args.use_corrected_flags)
    )
    m_joint_val = _metrics_from_scores(labs_concat_joint_val, preds_concat_joint_val)
    m_joint_test = _metrics_from_scores(labs_concat_joint_test, preds_concat_joint_test)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print(f"HLT baseline AUC (val/test): {m_hlt_val['auc']:.4f} / {m_hlt_test['auc']:.4f}")
    print(f"Concat pre-joint AUC (val/test): {m_pre_val['auc']:.4f} / {m_pre_test['auc']:.4f}")
    print(f"Concat post-joint AUC (val/test): {m_joint_val['auc']:.4f} / {m_joint_test['auc']:.4f}")
    print(
        "FPR@50 HLT / Concat(pre) / Concat(post): "
        f"{m_hlt_test['fpr50']:.6f} / {m_pre_test['fpr50']:.6f} / {m_joint_test['fpr50']:.6f}"
    )
    print(
        "FPR@30 HLT / Concat(pre) / Concat(post): "
        f"{m_hlt_test['fpr30']:.6f} / {m_pre_test['fpr30']:.6f} / {m_joint_test['fpr30']:.6f}"
    )

    if not bool(args.skip_save_models):
        torch.save({"model": reconstructor.state_dict(), "val": joint_metrics}, save_root / "offline_reconstructor_concat_stage2_joint.pt")
        torch.save({"model": concat_model.state_dict(), "val": joint_metrics}, save_root / "concat_singleview_joint.pt")
        if joint_states.get("fpr50", {}).get("model") is not None:
            torch.save(
                {"model": joint_states["fpr50"]["model"], "val": joint_metrics},
                save_root / "concat_singleview_joint_bestfpr50.pt",
            )
            torch.save(
                {"model": joint_states["fpr50"]["reco"], "val": joint_metrics},
                save_root / "offline_reconstructor_concat_stage2_joint_bestfpr50.pt",
            )

    np.savez_compressed(
        save_root / "concat_hltreco_stage2_results.npz",
        preds_hlt_val=preds_hlt_val.astype(np.float64),
        preds_hlt_test=preds_hlt_test.astype(np.float64),
        preds_concat_pre_val=preds_concat_pre_val.astype(np.float64),
        preds_concat_pre_test=preds_concat_pre_test.astype(np.float64),
        preds_concat_joint_val=preds_concat_joint_val.astype(np.float64),
        preds_concat_joint_test=preds_concat_joint_test.astype(np.float64),
        labs_val=labs_hlt_val.astype(np.float32),
        labs_test=labs_hlt_test.astype(np.float32),
    )

    report = {
        "run_dir": str(run_dir),
        "source_reco_ckpt": str(reco_ckpt),
        "source_baseline_ckpt": str(baseline_ckpt),
        "metrics": {
            "hlt_baseline": {"val": m_hlt_val, "test": m_hlt_test},
            "concat_pre_joint": {"val": m_pre_val, "test": m_pre_test},
            "concat_post_joint": {"val": m_joint_val, "test": m_joint_test},
        },
        "joint_selection": joint_metrics,
        "settings": {
            "corrected_weight_floor": float(args.corrected_weight_floor),
            "use_corrected_flags": bool(args.use_corrected_flags),
            "pretrain_epochs": int(args.pretrain_epochs),
            "joint_epochs": int(args.joint_epochs),
            "joint_lambda_reco": float(args.joint_lambda_reco),
            "joint_lambda_rank": float(args.joint_lambda_rank),
            "joint_lambda_cons": float(args.joint_lambda_cons),
            "selection_metric": str(args.selection_metric),
        },
    }
    with open(save_root / "concat_hltreco_stage2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if bool(args.save_fusion_scores):
        np.savez_compressed(
            save_root / "fusion_scores_val_test.npz",
            preds_hlt_val=preds_hlt_val.astype(np.float64),
            preds_hlt_test=preds_hlt_test.astype(np.float64),
            preds_joint_val=preds_concat_joint_val.astype(np.float64),
            preds_joint_test=preds_concat_joint_test.astype(np.float64),
            labels_val=labs_hlt_val.astype(np.float32),
            labels_test=labs_hlt_test.astype(np.float32),
        )
        print(f"Saved fusion score arrays to: {save_root / 'fusion_scores_val_test.npz'}")

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()
