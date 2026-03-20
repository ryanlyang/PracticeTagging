#!/usr/bin/env python3
"""
Stage-C continuation from saved Stage2 checkpoints, but with a reco-only top tagger.

What this script does:
1) Reloads the original data setup/splits deterministically from a saved run dir.
2) Loads Stage2 reconstructor checkpoint (typically offline_reconstructor_stage2.pt).
3) Trains a reco-only top tagger on soft corrected outputs with reconstructor frozen.
4) Saves frozen-phase selected checkpoint.
5) Continues joint finetuning by unfreezing reconstructor (same reco-only tagger).
6) Saves final selected checkpoint.

The reco-only classifier consumes only the reconstructed corrected view
(and optional merge/eff flags) -- no HLT branch / no dual-view classifier.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import offline_reconstructor_joint_dualview_stage2save_auc_norankc as joint
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as LOCAL30K_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
    compute_reconstruction_losses,
    fpr_at_target_tpr,
)
from unmerge_correct_hlt import (
    RANDOM_SEED,
    JetDataset,
    ParticleTransformer,
    compute_features,
    eval_classifier,
    get_scheduler,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_checkpoint_state(path: Path, device: torch.device, tag: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    keys = list(ckpt.keys())[:8] if isinstance(ckpt, dict) else [type(ckpt).__name__]
    raise RuntimeError(
        f"Unsupported checkpoint format for {tag}: {path}. "
        f"Top-level keys/type preview: {keys}"
    )


def load_cfg_from_run(run_dir: Path) -> Dict:
    cfg = joint._deepcopy_config()
    hlt_stats_path = run_dir / "hlt_stats.json"
    if hlt_stats_path.exists():
        h = json.load(open(hlt_stats_path, "r", encoding="utf-8"))
        hcfg = h.get("config", {})
        for k, v in hcfg.items():
            if k in cfg["hlt_effects"]:
                cfg["hlt_effects"][k] = v
    return cfg


def load_saved_data_setup(run_dir: Path) -> Dict:
    path = run_dir / "data_setup.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
        return out if isinstance(out, dict) else {}
    except Exception as e:
        print(f"Warning: failed to read saved data setup {path}: {e}")
        return {}


def load_saved_splits(run_dir: Path) -> Dict[str, np.ndarray]:
    path = run_dir / "data_splits.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as e:
        print(f"Warning: failed to read saved splits {path}: {e}")
        return {}


class RecoOnlyStageCDataset(Dataset):
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
        return int(self.feat_hlt_reco.shape[0])

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
def eval_recoonly_joint_model(
    reconstructor: OfflineReconstructor,
    reco_clf: nn.Module,
    loader: DataLoader,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    reconstructor.eval()
    reco_clf.eval()
    preds: List[np.ndarray] = []
    labs: List[np.ndarray] = []

    for batch in loader:
        feat_hlt_reco = batch["feat_hlt_reco"].to(device)
        mask_hlt = batch["mask_hlt"].to(device)
        const_hlt = batch["const_hlt"].to(device)
        y = batch["label"].detach().cpu().numpy().astype(np.float32)

        reco_out = reconstructor(feat_hlt_reco, mask_hlt, const_hlt, stage_scale=1.0)
        feat_b, mask_b = joint.build_soft_corrected_view(
            reco_out,
            weight_floor=float(corrected_weight_floor),
            scale_features_by_weight=True,
            include_flags=bool(corrected_use_flags),
        )
        logits = reco_clf(feat_b, mask_b).squeeze(1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        preds.append(p)
        labs.append(y)

    if len(preds) == 0:
        return float("nan"), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), float("nan")

    pred = np.concatenate(preds).astype(np.float32)
    lab = np.concatenate(labs).astype(np.int64)
    if len(np.unique(lab)) < 2:
        return float("nan"), pred, lab, float("nan")

    fpr, tpr, _ = roc_curve(lab, pred)
    auc = float(np.trapz(tpr, fpr))
    fpr50 = float(fpr_at_target_tpr(fpr, tpr, 0.50))
    return float(auc), pred, lab, float(fpr50)


def train_recoonly_joint(
    reconstructor: OfflineReconstructor,
    reco_clf: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage_name: str,
    freeze_reconstructor: bool,
    epochs: int,
    patience: int,
    lr_cls: float,
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
) -> Tuple[OfflineReconstructor, nn.Module, Dict[str, float], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    for p in reconstructor.parameters():
        p.requires_grad = not freeze_reconstructor

    params = [{"params": reco_clf.parameters(), "lr": float(lr_cls)}]
    if not freeze_reconstructor:
        params.append({"params": reconstructor.parameters(), "lr": float(lr_reco)})

    opt = torch.optim.AdamW(params, lr=float(lr_cls), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))

    best_state_cls_sel = None
    best_state_reco_sel = None
    best_state_cls_auc = None
    best_state_reco_auc = None
    best_state_cls_fpr = None
    best_state_reco_fpr = None

    best_val_fpr50 = float("inf")
    best_val_auc = float("-inf")
    best_sel_score = float("inf") if str(select_metric).lower() == "fpr50" else float("-inf")
    sel_val_fpr50 = float("nan")
    sel_val_auc = float("nan")
    no_improve = 0

    for ep in tqdm(range(int(epochs)), desc=stage_name):
        reco_clf.train()
        if freeze_reconstructor:
            reconstructor.eval()
        else:
            reconstructor.train()

        tr_loss = 0.0
        tr_cls = 0.0
        tr_rank = 0.0
        tr_reco = 0.0
        tr_cons = 0.0
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

            feat_b, mask_b = joint.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            logits = reco_clf(feat_b, mask_b).squeeze(1)

            loss_cls = F.binary_cross_entropy_with_logits(logits, y)
            loss_rank = joint.low_fpr_surrogate_loss(logits, y, target_tpr=0.50, tau=0.05)
            loss_cons = reco_out["child_weight"].mean() + reco_out["gen_weight"].mean()

            if float(lambda_reco) > 0.0:
                reco_losses = compute_reconstruction_losses(
                    reco_out,
                    const_hlt,
                    mask_hlt,
                    const_off,
                    mask_off,
                    b_merge,
                    b_eff,
                    LOCAL30K_CONFIG["loss"],
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
            torch.nn.utils.clip_grad_norm_(reco_clf.parameters(), 1.0)
            if not freeze_reconstructor:
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0)
            opt.step()

            bs = int(feat_hlt_reco.size(0))
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

        va_auc, _, _, va_fpr50 = eval_recoonly_joint_model(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            loader=val_loader,
            device=device,
            corrected_weight_floor=float(corrected_weight_floor),
            corrected_use_flags=bool(corrected_use_flags),
        )

        if np.isfinite(va_fpr50) and float(va_fpr50) < best_val_fpr50:
            best_val_fpr50 = float(va_fpr50)
            best_state_cls_fpr = {k: v.detach().cpu().clone() for k, v in reco_clf.state_dict().items()}
            best_state_reco_fpr = {k: v.detach().cpu().clone() for k, v in reconstructor.state_dict().items()}
        if np.isfinite(va_auc) and float(va_auc) > best_val_auc:
            best_val_auc = float(va_auc)
            best_state_cls_auc = {k: v.detach().cpu().clone() for k, v in reco_clf.state_dict().items()}
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
            best_state_cls_sel = {k: v.detach().cpu().clone() for k, v in reco_clf.state_dict().items()}
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

    if best_state_cls_sel is not None:
        reco_clf.load_state_dict(best_state_cls_sel)
    if best_state_reco_sel is not None:
        reconstructor.load_state_dict(best_state_reco_sel)

    metrics = {
        "selection_metric": str(select_metric).lower(),
        "selected_val_fpr50": float(sel_val_fpr50),
        "selected_val_auc": float(sel_val_auc),
        "best_val_fpr50_seen": float(best_val_fpr50),
        "best_val_auc_seen": float(best_val_auc),
    }
    state_pack = {
        "selected": {"clf": best_state_cls_sel, "reco": best_state_reco_sel},
        "auc": {"clf": best_state_cls_auc, "reco": best_state_reco_auc},
        "fpr50": {"clf": best_state_cls_fpr, "reco": best_state_reco_fpr},
    }
    return reconstructor, reco_clf, metrics, state_pack


def maybe_eval_single_view_checkpoint(
    ckpt_path: Path,
    tag: str,
    feat_test: np.ndarray,
    mask_test: np.ndarray,
    labels_test: np.ndarray,
    model_cfg: Dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, float]:
    if not ckpt_path.exists():
        print(f"Warning: {tag} checkpoint not found: {ckpt_path}")
        return {}

    model = ParticleTransformer(input_dim=7, **model_cfg).to(device)
    state = _load_checkpoint_state(ckpt_path, device, tag)
    model.load_state_dict(state)

    ds = JetDataset(feat_test, mask_test, labels_test)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    auc, preds, labs = eval_classifier(model, dl, device)
    fpr, tpr, _ = roc_curve(labs, preds)
    return {
        "auc": float(auc),
        "fpr30": float(fpr_at_target_tpr(fpr, tpr, 0.30)),
        "fpr50": float(fpr_at_target_tpr(fpr, tpr, 0.50)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Previous run folder with Stage2 checkpoints")
    p.add_argument("--save_dir", type=str, default="", help="If empty, defaults to <run_dir>/stagec_refine")
    p.add_argument("--run_name", type=str, default="stagec_recoonly_refine")

    p.add_argument("--train_path", type=str, default="./data")
    p.add_argument("--n_train_jets", type=int, default=100000)
    p.add_argument("--offset_jets", type=int, default=0)
    p.add_argument("--max_constits", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=-1)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--ignore_saved_data_setup",
        action="store_true",
        help="Ignore run_dir/data_setup.json and run_dir/data_splits.npz; rebuild from CLI args.",
    )

    p.add_argument("--reco_ckpt", type=str, default="")

    # Stage C schedule and loss settings.
    p.add_argument("--stageC_epochs", type=int, default=70)
    p.add_argument("--stageC_patience", type=int, default=12)
    p.add_argument("--stageC_min_epochs", type=int, default=25)
    p.add_argument("--stageC_freeze_reco_epochs", type=int, default=20)
    p.add_argument("--stageC_lr_cls", type=float, default=1e-5)
    p.add_argument("--stageC_lr_reco", type=float, default=5e-6)
    p.add_argument("--stageC_lambda_rank", type=float, default=0.0)
    p.add_argument("--lambda_reco", type=float, default=0.4)
    p.add_argument("--lambda_cons", type=float, default=0.06)
    p.add_argument("--selection_metric", type=str, default="auc", choices=["auc", "fpr50"])
    p.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    p.add_argument("--use_corrected_flags", action="store_true")

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    saved_setup = {}
    saved_splits = {}
    use_saved_data_setup = False
    if not bool(args.ignore_saved_data_setup):
        saved_setup = load_saved_data_setup(run_dir)
        saved_splits = load_saved_splits(run_dir)
        use_saved_data_setup = len(saved_setup) > 0

    eff_seed = int(saved_setup.get("seed", args.seed)) if use_saved_data_setup else int(args.seed)
    eff_n_train_jets = int(saved_setup.get("n_train_jets", args.n_train_jets)) if use_saved_data_setup else int(args.n_train_jets)
    eff_offset_jets = int(saved_setup.get("offset_jets", args.offset_jets)) if use_saved_data_setup else int(args.offset_jets)
    eff_max_constits = int(saved_setup.get("max_constits", args.max_constits)) if use_saved_data_setup else int(args.max_constits)
    set_seed(eff_seed)

    out_root = Path(args.save_dir) if str(args.save_dir).strip() else (run_dir / "stagec_refine")
    save_root = out_root / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg_from_run(run_dir)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Load run dir: {run_dir}")
    print(f"Save dir: {save_root}")

    train_path = Path(args.train_path)
    train_files_saved = saved_setup.get("train_files", None) if use_saved_data_setup else None
    if isinstance(train_files_saved, list) and len(train_files_saved) > 0:
        train_files = [Path(x) for x in train_files_saved]
    else:
        train_files = [train_path / "test.h5"]

    source_tag = "saved data_setup.json" if use_saved_data_setup else "CLI args"
    print(
        f"Data setup source: {source_tag} | "
        f"seed={eff_seed}, n_train_jets={eff_n_train_jets}, offset_jets={eff_offset_jets}, max_constits={eff_max_constits}"
    )

    print("Loading offline constituents...")
    const_raw, labels = load_raw_constituents_from_h5(
        train_files,
        max_jets=eff_n_train_jets,
        max_constits=eff_max_constits,
        offset=eff_offset_jets,
    )

    masks_off = const_raw[:, :, 0] > 0.0
    const_off = const_raw.copy()

    print("Generating pseudo-HLT deterministically...")
    hlt_const, hlt_mask, hlt_stats, budget_truth = apply_hlt_effects_realistic_nomap(
        const_off,
        masks_off,
        cfg,
        seed=eff_seed,
    )

    print("Computing features...")
    feat_off = compute_features(const_off, masks_off)
    feat_hlt = compute_features(hlt_const, hlt_mask)

    saved_train_idx = saved_splits.get("train_idx") if isinstance(saved_splits, dict) else None
    saved_val_idx = saved_splits.get("val_idx") if isinstance(saved_splits, dict) else None
    saved_test_idx = saved_splits.get("test_idx") if isinstance(saved_splits, dict) else None
    has_saved_split_idx = (
        saved_train_idx is not None
        and saved_val_idx is not None
        and saved_test_idx is not None
    )

    if use_saved_data_setup and has_saved_split_idx:
        train_idx = np.asarray(saved_train_idx, dtype=np.int64)
        val_idx = np.asarray(saved_val_idx, dtype=np.int64)
        test_idx = np.asarray(saved_test_idx, dtype=np.int64)
        all_idx = np.arange(labels.shape[0], dtype=np.int64)
        used = np.concatenate([train_idx, val_idx, test_idx], axis=0)
        if np.any(used < 0) or np.any(used >= labels.shape[0]):
            raise ValueError("Saved split indices are out of bounds for currently loaded data.")
        if np.unique(used).shape[0] != used.shape[0]:
            raise ValueError("Saved split indices contain duplicates across train/val/test.")
        missing = np.setdiff1d(all_idx, used)
        if missing.size > 0:
            print(f"Warning: saved splits do not cover {missing.size} samples; they will be ignored.")
        splits_source = "saved data_splits.npz"
    else:
        idx = np.arange(labels.shape[0])
        train_idx, tmp_idx = train_test_split(
            idx,
            test_size=0.30,
            random_state=eff_seed,
            stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=0.50,
            random_state=eff_seed,
            stratify=labels[tmp_idx],
        )
        splits_source = "fresh train_test_split"

    if use_saved_data_setup and isinstance(saved_splits, dict) and "means" in saved_splits and "stds" in saved_splits:
        means = np.asarray(saved_splits["means"], dtype=np.float32)
        stds = np.asarray(saved_splits["stds"], dtype=np.float32)
    else:
        means, stds = get_stats(feat_hlt[train_idx], hlt_mask[train_idx])

    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds)
    feat_off_std = standardize(feat_off, masks_off, means, stds)

    true_count = masks_off.sum(axis=1).astype(np.float32)
    hlt_count = hlt_mask.sum(axis=1).astype(np.float32)
    true_added_raw = np.maximum(true_count - hlt_count, 0.0).astype(np.float32)

    stage_metrics_path = run_dir / "joint_stage_metrics.json"
    added_target_scale = 1.0
    if stage_metrics_path.exists():
        try:
            m = json.load(open(stage_metrics_path, "r", encoding="utf-8"))
            added_target_scale = float(m.get("variant", {}).get("added_target_scale", 1.0))
        except Exception:
            pass

    budget_merge_true = (added_target_scale * true_added_raw).astype(np.float32)
    budget_eff_true = np.zeros_like(true_added_raw, dtype=np.float32)

    bs = int(cfg["training"]["batch_size"]) if int(args.batch_size) <= 0 else int(args.batch_size)

    ds_train = RecoOnlyStageCDataset(
        feat_hlt_reco=feat_hlt_std[train_idx],
        mask_hlt=hlt_mask[train_idx],
        const_hlt=hlt_const[train_idx],
        const_off=const_off[train_idx],
        mask_off=masks_off[train_idx],
        budget_merge_true=budget_merge_true[train_idx],
        budget_eff_true=budget_eff_true[train_idx],
        labels=labels[train_idx],
    )
    ds_val = RecoOnlyStageCDataset(
        feat_hlt_reco=feat_hlt_std[val_idx],
        mask_hlt=hlt_mask[val_idx],
        const_hlt=hlt_const[val_idx],
        const_off=const_off[val_idx],
        mask_off=masks_off[val_idx],
        budget_merge_true=budget_merge_true[val_idx],
        budget_eff_true=budget_eff_true[val_idx],
        labels=labels[val_idx],
    )
    ds_test = RecoOnlyStageCDataset(
        feat_hlt_reco=feat_hlt_std[test_idx],
        mask_hlt=hlt_mask[test_idx],
        const_hlt=hlt_const[test_idx],
        const_off=const_off[test_idx],
        mask_off=masks_off[test_idx],
        budget_merge_true=budget_merge_true[test_idx],
        budget_eff_true=budget_eff_true[test_idx],
        labels=labels[test_idx],
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} "
        f"(source: {splits_source})"
    )

    reco_ckpt = Path(args.reco_ckpt) if str(args.reco_ckpt).strip() else (run_dir / "offline_reconstructor_stage2.pt")
    if not reco_ckpt.exists():
        raise FileNotFoundError(
            f"Stage2 reconstructor checkpoint not found: {reco_ckpt}. "
            "Expected offline_reconstructor_stage2.pt in run_dir or pass --reco_ckpt."
        )

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(_load_checkpoint_state(reco_ckpt, device, "stage2_reconstructor"))

    reco_input_dim = 12 if bool(args.use_corrected_flags) else 10
    reco_clf = ParticleTransformer(input_dim=reco_input_dim, **cfg["model"]).to(device)

    teacher_metrics = maybe_eval_single_view_checkpoint(
        ckpt_path=run_dir / "teacher.pt",
        tag="teacher",
        feat_test=feat_off_std[test_idx],
        mask_test=masks_off[test_idx],
        labels_test=labels[test_idx],
        model_cfg=cfg["model"],
        batch_size=bs,
        num_workers=int(args.num_workers),
        device=device,
    )
    baseline_metrics = maybe_eval_single_view_checkpoint(
        ckpt_path=run_dir / "baseline.pt",
        tag="baseline",
        feat_test=feat_hlt_std[test_idx],
        mask_test=hlt_mask[test_idx],
        labels_test=labels[test_idx],
        model_cfg=cfg["model"],
        batch_size=bs,
        num_workers=int(args.num_workers),
        device=device,
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C (RECO-ONLY): FROZEN -> UNFREEZE")
    print("=" * 70)

    LOCAL30K_CONFIG["loss"] = cfg["loss"]
    total_stagec_epochs = int(args.stageC_epochs)
    freeze_epochs = max(0, min(int(args.stageC_freeze_reco_epochs), total_stagec_epochs))
    unfreeze_epochs = max(0, total_stagec_epochs - freeze_epochs)
    if freeze_epochs > 0:
        print(
            f"Stage-C schedule: freeze reconstructor for {freeze_epochs} epoch(s), "
            f"then unfreeze for {unfreeze_epochs} epoch(s)."
        )
    else:
        print("Stage-C schedule: reconstructor unfrozen from epoch 1.")

    def _is_auc_mode() -> bool:
        return str(args.selection_metric).lower() == "auc"

    def _better_selected(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        if _is_auc_mode():
            return float(new_m.get("selected_val_auc", float("-inf"))) > float(cur_m.get("selected_val_auc", float("-inf")))
        return float(new_m.get("selected_val_fpr50", float("inf"))) < float(cur_m.get("selected_val_fpr50", float("inf")))

    def _better_auc(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        return float(new_m.get("best_val_auc_seen", float("-inf"))) > float(cur_m.get("best_val_auc_seen", float("-inf")))

    def _better_fpr(new_m: Dict[str, float], cur_m: Dict[str, float] | None) -> bool:
        if cur_m is None:
            return True
        return float(new_m.get("best_val_fpr50_seen", float("inf"))) < float(cur_m.get("best_val_fpr50_seen", float("inf")))

    selected_metrics = None
    auc_metrics = None
    fpr_metrics = None
    selected_states = None
    auc_states = None
    fpr_states = None
    frozen_selected_metrics = None
    frozen_selected_states = None
    phase_reports = []

    def _run_phase(phase_name: str, freeze_reco: bool, epochs: int, patience: int, min_epochs: int) -> None:
        nonlocal reconstructor, reco_clf
        nonlocal selected_metrics, auc_metrics, fpr_metrics
        nonlocal selected_states, auc_states, fpr_states
        nonlocal frozen_selected_metrics, frozen_selected_states
        if int(epochs) <= 0:
            return
        reconstructor, reco_clf, ph_metrics, ph_states = train_recoonly_joint(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            train_loader=dl_train,
            val_loader=dl_val,
            device=device,
            stage_name=phase_name,
            freeze_reconstructor=bool(freeze_reco),
            epochs=int(epochs),
            patience=int(patience),
            lr_cls=float(args.stageC_lr_cls),
            lr_reco=float(args.stageC_lr_reco),
            weight_decay=float(cfg["training"]["weight_decay"]),
            warmup_epochs=int(cfg["training"]["warmup_epochs"]),
            lambda_reco=float(args.lambda_reco),
            lambda_rank=float(args.stageC_lambda_rank),
            lambda_cons=float(args.lambda_cons),
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
            min_epochs=int(min_epochs),
            select_metric=str(args.selection_metric),
        )
        phase_reports.append(
            {
                "phase_name": phase_name,
                "freeze_reconstructor": bool(freeze_reco),
                "epochs": int(epochs),
                "metrics": ph_metrics,
            }
        )
        if _better_selected(ph_metrics, selected_metrics):
            selected_metrics = ph_metrics
            selected_states = ph_states.get("selected", {})
        if _better_auc(ph_metrics, auc_metrics):
            auc_metrics = ph_metrics
            auc_states = ph_states.get("auc", {})
        if _better_fpr(ph_metrics, fpr_metrics):
            fpr_metrics = ph_metrics
            fpr_states = ph_states.get("fpr50", {})
        if bool(freeze_reco):
            frozen_selected_metrics = ph_metrics
            frozen_selected_states = ph_states.get("selected", {})

    if freeze_epochs > 0:
        _run_phase(
            phase_name="StageC-RecoOnly-FrozenReco",
            freeze_reco=True,
            epochs=int(freeze_epochs),
            patience=max(int(freeze_epochs) + 1, int(args.stageC_patience)),
            min_epochs=int(freeze_epochs),
        )
        if (frozen_selected_states or {}).get("reco") is not None:
            reconstructor.load_state_dict(frozen_selected_states["reco"])
        if (frozen_selected_states or {}).get("clf") is not None:
            reco_clf.load_state_dict(frozen_selected_states["clf"])

    _run_phase(
        phase_name="StageC-RecoOnly",
        freeze_reco=False,
        epochs=int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs),
        patience=int(args.stageC_patience),
        min_epochs=min(int(args.stageC_min_epochs), int(unfreeze_epochs if freeze_epochs > 0 else total_stagec_epochs)),
    )

    stageC_metrics = {
        "selection_metric": str(args.selection_metric).lower(),
        "selected_val_fpr50": float(selected_metrics.get("selected_val_fpr50", float("nan"))) if selected_metrics else float("nan"),
        "selected_val_auc": float(selected_metrics.get("selected_val_auc", float("nan"))) if selected_metrics else float("nan"),
        "best_val_fpr50_seen": float(fpr_metrics.get("best_val_fpr50_seen", float("nan"))) if fpr_metrics else float("nan"),
        "best_val_auc_seen": float(auc_metrics.get("best_val_auc_seen", float("nan"))) if auc_metrics else float("nan"),
    }
    stageC_states = {
        "selected": {"clf": (selected_states or {}).get("clf"), "reco": (selected_states or {}).get("reco")},
        "auc": {"clf": (auc_states or {}).get("clf"), "reco": (auc_states or {}).get("reco")},
        "fpr50": {"clf": (fpr_states or {}).get("clf"), "reco": (fpr_states or {}).get("reco")},
        "frozen_selected": {"clf": (frozen_selected_states or {}).get("clf"), "reco": (frozen_selected_states or {}).get("reco")},
        "phase_reports": phase_reports,
    }

    # Evaluate frozen selected (if present).
    auc_frozen = float("nan")
    fpr30_frozen = float("nan")
    fpr50_frozen = float("nan")
    preds_frozen = np.array([], dtype=np.float32)
    labs_ref = None
    if stageC_states.get("frozen_selected", {}).get("clf") is not None and stageC_states.get("frozen_selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["frozen_selected"]["reco"])
        reco_clf.load_state_dict(stageC_states["frozen_selected"]["clf"])
        torch.save({"model": reconstructor.state_dict()}, save_root / "offline_reconstructor_recoonly_stagec_frozen_ckpt.pt")
        torch.save({"model": reco_clf.state_dict()}, save_root / "recoonly_classifier_stagec_frozen_ckpt.pt")
        auc_frozen, preds_frozen, labs_frozen, _ = eval_recoonly_joint_model(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            loader=dl_test,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        if len(labs_frozen) > 0 and np.isfinite(auc_frozen):
            fpr_fr, tpr_fr, _ = roc_curve(labs_frozen, preds_frozen)
            fpr30_frozen = float(fpr_at_target_tpr(fpr_fr, tpr_fr, 0.30))
            fpr50_frozen = float(fpr_at_target_tpr(fpr_fr, tpr_fr, 0.50))
        labs_ref = labs_frozen

    # Evaluate selected checkpoint.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("clf") is not None:
        reco_clf.load_state_dict(stageC_states["selected"]["clf"])

    auc_selected, preds_selected, labs_selected, _ = eval_recoonly_joint_model(
        reconstructor=reconstructor,
        reco_clf=reco_clf,
        loader=dl_test,
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=bool(args.use_corrected_flags),
    )
    fpr30_selected = float("nan")
    fpr50_selected = float("nan")
    if len(labs_selected) > 0 and np.isfinite(auc_selected):
        fpr_sel, tpr_sel, _ = roc_curve(labs_selected, preds_selected)
        fpr30_selected = float(fpr_at_target_tpr(fpr_sel, tpr_sel, 0.30))
        fpr50_selected = float(fpr_at_target_tpr(fpr_sel, tpr_sel, 0.50))
    labs_ref = labs_selected if labs_ref is None else labs_ref

    # Evaluate best-val-fpr50 checkpoint for reference.
    auc_bestfpr = float("nan")
    fpr30_bestfpr = float("nan")
    fpr50_bestfpr = float("nan")
    if stageC_states.get("fpr50", {}).get("clf") is not None and stageC_states.get("fpr50", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["fpr50"]["reco"])
        reco_clf.load_state_dict(stageC_states["fpr50"]["clf"])
        auc_bestfpr, preds_bestfpr, labs_bestfpr, _ = eval_recoonly_joint_model(
            reconstructor=reconstructor,
            reco_clf=reco_clf,
            loader=dl_test,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
            corrected_use_flags=bool(args.use_corrected_flags),
        )
        if len(labs_bestfpr) > 0 and np.isfinite(auc_bestfpr):
            fpr_bf, tpr_bf, _ = roc_curve(labs_bestfpr, preds_bestfpr)
            fpr30_bestfpr = float(fpr_at_target_tpr(fpr_bf, tpr_bf, 0.30))
            fpr50_bestfpr = float(fpr_at_target_tpr(fpr_bf, tpr_bf, 0.50))

    # Restore selected for saving.
    if stageC_states.get("selected", {}).get("reco") is not None:
        reconstructor.load_state_dict(stageC_states["selected"]["reco"])
    if stageC_states.get("selected", {}).get("clf") is not None:
        reco_clf.load_state_dict(stageC_states["selected"]["clf"])

    torch.save({"model": reconstructor.state_dict()}, save_root / "offline_reconstructor_recoonly_stagec_selected_ckpt.pt")
    torch.save({"model": reco_clf.state_dict()}, save_root / "recoonly_classifier_stagec_selected_ckpt.pt")
    torch.save(reconstructor.state_dict(), save_root / "offline_reconstructor.pt")
    torch.save(reco_clf.state_dict(), save_root / "recoonly_classifier.pt")

    for fname in ["data_setup.json", "data_splits.npz", "teacher.pt", "baseline.pt", "hlt_stats.json"]:
        src = run_dir / fname
        if src.exists():
            try:
                shutil.copy2(src, save_root / fname)
            except Exception as e:
                print(f"Warning: failed to copy {src} -> {save_root / fname}: {e}")

    out_metrics = {
        "source_run_dir": str(run_dir),
        "source_reco_ckpt": str(reco_ckpt),
        "stageC_args": {
            "stageC_epochs": int(args.stageC_epochs),
            "stageC_patience": int(args.stageC_patience),
            "stageC_min_epochs": int(args.stageC_min_epochs),
            "stageC_freeze_reco_epochs": int(args.stageC_freeze_reco_epochs),
            "stageC_lr_cls": float(args.stageC_lr_cls),
            "stageC_lr_reco": float(args.stageC_lr_reco),
            "stageC_lambda_rank": float(args.stageC_lambda_rank),
            "lambda_reco": float(args.lambda_reco),
            "lambda_cons": float(args.lambda_cons),
            "selection_metric": str(args.selection_metric),
            "corrected_weight_floor": float(args.corrected_weight_floor),
            "use_corrected_flags": bool(args.use_corrected_flags),
            "recoonly_input_dim": int(reco_input_dim),
        },
        "data_reload": {
            "setup_source": "saved data_setup.json" if use_saved_data_setup else "cli args",
            "splits_source": splits_source,
            "seed_effective": int(eff_seed),
            "n_train_jets_effective": int(eff_n_train_jets),
            "offset_jets_effective": int(eff_offset_jets),
            "max_constits_effective": int(eff_max_constits),
            "train_files_used": [str(pth) for pth in train_files],
            "ignore_saved_data_setup": bool(args.ignore_saved_data_setup),
        },
        "stageC_metrics": stageC_metrics,
        "stageC_phase_reports": phase_reports,
        "test_stageC_frozen_selected": {
            "auc": float(auc_frozen),
            "fpr30": float(fpr30_frozen),
            "fpr50": float(fpr50_frozen),
        },
        "test_stageC_selected": {
            "auc": float(auc_selected),
            "fpr30": float(fpr30_selected),
            "fpr50": float(fpr50_selected),
        },
        "test_stageC_bestfpr50": {
            "auc": float(auc_bestfpr),
            "fpr30": float(fpr30_bestfpr),
            "fpr50": float(fpr50_bestfpr),
        },
        "test_teacher_loaded": teacher_metrics,
        "test_baseline_loaded": baseline_metrics,
    }

    with open(save_root / "stagec_recoonly_refine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    np.savez_compressed(
        save_root / "results_recoonly.npz",
        labels=(labs_ref.astype(np.float32) if labs_ref is not None else np.array([], dtype=np.float32)),
        preds_stagec_frozen=preds_frozen.astype(np.float32),
        preds_stagec_selected=preds_selected.astype(np.float32),
    )

    print("\n" + "=" * 70)
    print("FAST STAGE C RECO-ONLY RESULTS")
    print("=" * 70)
    if len(teacher_metrics) > 0:
        print(
            f"Teacher (loaded): AUC={teacher_metrics['auc']:.4f}, "
            f"FPR30={teacher_metrics['fpr30']:.6f}, FPR50={teacher_metrics['fpr50']:.6f}"
        )
    if len(baseline_metrics) > 0:
        print(
            f"Baseline (loaded): AUC={baseline_metrics['auc']:.4f}, "
            f"FPR30={baseline_metrics['fpr30']:.6f}, FPR50={baseline_metrics['fpr50']:.6f}"
        )
    if np.isfinite(auc_frozen):
        print(f"RecoOnly FrozenSelected: AUC={auc_frozen:.4f}, FPR30={fpr30_frozen:.6f}, FPR50={fpr50_frozen:.6f}")
    print(f"RecoOnly Selected: AUC={auc_selected:.4f}, FPR30={fpr30_selected:.6f}, FPR50={fpr50_selected:.6f}")
    if np.isfinite(auc_bestfpr):
        print(
            f"RecoOnly BestValFPR50: AUC={auc_bestfpr:.4f}, "
            f"FPR30={fpr30_bestfpr:.6f}, FPR50={fpr50_bestfpr:.6f}"
        )
    print(f"Saved to: {save_root}")


if __name__ == "__main__":
    main()
