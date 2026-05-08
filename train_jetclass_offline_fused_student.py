#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an offline student classifier to match fused soft targets.

Inputs:
- fused target bundle from build_jetclass_fused_targets_two_model.py
- deterministic split rebuild from the reference run args
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_jetclass_hlt_teacher_baseline import (
    CANONICAL_CLASS_ORDER,
    JetClassTransformer,
    collect_files_by_class,
    compute_features,
    eval_metrics,
    get_mean_std,
    load_split,
    make_loader,
    set_seed,
    split_files_by_class,
    standardize,
)


class DistillDataset(Dataset):
    def __init__(
        self,
        feat: np.ndarray,
        mask: np.ndarray,
        label: np.ndarray,
        target_probs: np.ndarray,
    ):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.label = torch.tensor(label.astype(np.int64), dtype=torch.long)
        self.target = torch.tensor(target_probs, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat": self.feat[i],
            "mask": self.mask[i],
            "label": self.label[i],
            "target": self.target[i],
        }


def _ns_from_json(path: Path) -> SimpleNamespace:
    return SimpleNamespace(**json.loads(path.read_text()))


def _resolve_run_ref_dir(targets_meta: Dict[str, object], cli_run_ref: Path | None) -> Path:
    if cli_run_ref is not None:
        return cli_run_ref.resolve()
    p = targets_meta.get("run_ref_dir", None)
    if p is None:
        raise ValueError("run_ref_dir missing from metadata; pass --run_ref_dir explicitly.")
    return Path(str(p)).resolve()


def _build_offline_kin_features(
    args_ref: SimpleNamespace,
    data_dir: Path,
) -> Tuple[
    Sequence[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    files_by_class = collect_files_by_class(data_dir.resolve())
    if str(args_ref.class_assignment) == "canonical_labels":
        class_names = list(CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=int(args_ref.train_files_per_class),
        n_val=int(args_ref.val_files_per_class),
        n_test=int(args_ref.test_files_per_class),
        shuffle=bool(args_ref.shuffle_files),
        seed=int(args_ref.seed),
    )

    tr_tok_raw, tr_mask_raw, tr_y = load_split(
        tr_files,
        n_total=int(args_ref.n_train_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 101,
        class_assignment=str(args_ref.class_assignment),
    )
    va_tok_raw, va_mask_raw, va_y = load_split(
        va_files,
        n_total=int(args_ref.n_val_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 202,
        class_assignment=str(args_ref.class_assignment),
    )
    te_tok_raw, te_mask_raw, te_y = load_split(
        te_files,
        n_total=int(args_ref.n_test_jets),
        max_constits=int(args_ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args_ref.seed) + 303,
        class_assignment=str(args_ref.class_assignment),
    )

    tr_feat = compute_features(tr_tok_raw, tr_mask_raw, feature_mode="kin", feature_preprocessing="canonical")
    va_feat = compute_features(va_tok_raw, va_mask_raw, feature_mode="kin", feature_preprocessing="canonical")
    te_feat = compute_features(te_tok_raw, te_mask_raw, feature_mode="kin", feature_preprocessing="canonical")

    # Keep canonical (fixed) scaling by default.
    if str(getattr(args_ref, "feature_preprocessing", "canonical")) != "canonical":
        idx_all = np.arange(len(tr_y))
        mean, std = get_mean_std(tr_feat, tr_mask_raw, idx_all)
        tr_feat = standardize(tr_feat, tr_mask_raw, mean, std)
        va_feat = standardize(va_feat, va_mask_raw, mean, std)
        te_feat = standardize(te_feat, te_mask_raw, mean, std)

    return (
        class_names,
        tr_feat.astype(np.float32),
        tr_mask_raw.astype(bool),
        tr_y.astype(np.int64),
        va_feat.astype(np.float32),
        va_mask_raw.astype(bool),
        va_y.astype(np.int64),
        te_feat.astype(np.float32),
        te_mask_raw.astype(bool),
        te_y.astype(np.int64),
    )


def _scheduler(opt: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(ep: int) -> float:
        if ep < int(warmup_epochs):
            return (ep + 1) / max(int(warmup_epochs), 1)
        x = (ep - int(warmup_epochs)) / max(int(total_epochs) - int(warmup_epochs), 1)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def _eval_student(
    model: JetClassTransformer,
    loader: DataLoader,
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    temp: float,
    lambda_kl: float,
    lambda_ce: float,
    use_conf_weight: bool,
) -> Dict[str, object]:
    model.eval()
    n_cls = int(len(class_names))
    total = total_kl = total_ce = 0.0
    n = 0
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for batch in loader:
        x = batch["feat"].to(next(model.parameters()).device)
        m = batch["mask"].to(next(model.parameters()).device)
        y = batch["label"].to(next(model.parameters()).device)
        p_t = batch["target"].to(next(model.parameters()).device)

        logits = model(x, m)
        q_log = F.log_softmax(logits / float(temp), dim=1)
        p_norm = torch.clamp(p_t, min=1e-8)
        p_norm = p_norm / p_norm.sum(dim=1, keepdim=True)
        kl_vec = torch.sum(p_norm * (torch.log(p_norm) - q_log), dim=1) * (float(temp) ** 2)

        if bool(use_conf_weight):
            ent = -torch.sum(p_norm * torch.log(p_norm), dim=1)
            conf = 1.0 - ent / math.log(max(n_cls, 2))
            conf = torch.clamp(conf, min=0.05, max=1.0)
            loss_kl = (conf * kl_vec).mean()
        else:
            loss_kl = kl_vec.mean()

        loss_ce = F.cross_entropy(logits, y)
        loss = float(lambda_kl) * loss_kl + float(lambda_ce) * loss_ce

        bs = int(y.shape[0])
        total += float(loss.item()) * bs
        total_kl += float(loss_kl.item()) * bs
        total_ce += float(loss_ce.item()) * bs
        n += bs

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, n_cls), dtype=np.float32)
    ys = np.concatenate(all_y, axis=0) if all_y else np.zeros((0,), dtype=np.int64)

    out: Dict[str, object] = {
        "loss": total / max(n, 1),
        "loss_kl": total_kl / max(n, 1),
        "loss_ce": total_ce / max(n, 1),
        "probs": probs,
        "labels": ys,
    }
    if ys.size == 0:
        return out
    out.update(eval_metrics(ys, probs, class_names, background_class, target_class))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train offline student on fused soft targets")
    ap.add_argument("--targets_dir", type=Path, required=True)
    ap.add_argument("--run_ref_dir", type=Path, default=None)
    ap.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    ap.add_argument("--save_dir", type=Path, default=Path("checkpoints/jetclass_joint_dualview"))
    ap.add_argument("--run_name", type=str, default="jetclass_offline_fused_student_50k25k100k")
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=4)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--ff_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--target_key", type=str, default="probs_fused_bin", choices=["probs_fused_bin", "probs_fused_global"])
    ap.add_argument("--distill_temp", type=float, default=2.5)
    ap.add_argument("--lambda_kl", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=0.08)
    ap.add_argument("--use_conf_weight", action="store_true")
    args = ap.parse_args()

    set_seed(int(args.seed))
    save_root = (args.save_dir / args.run_name).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")

    targets_dir = args.targets_dir.resolve()
    arr = np.load(targets_dir / "fused_targets_train_val_test.npz")
    meta = json.loads((targets_dir / "fused_targets_metadata.json").read_text())
    run_ref_dir = _resolve_run_ref_dir(meta, args.run_ref_dir)
    args_ref = _ns_from_json(run_ref_dir / "args.json")

    class_names = list(meta["class_names"])
    target_class = str(meta["target_class"])
    background_class = str(meta["background_class"])

    (
        class_names_ref,
        tr_feat,
        tr_mask,
        tr_y,
        va_feat,
        va_mask,
        va_y,
        te_feat,
        te_mask,
        te_y,
    ) = _build_offline_kin_features(args_ref, args.data_dir)

    if list(class_names_ref) != list(class_names):
        raise ValueError("Class name mismatch between reference rebuild and fused-target metadata.")
    if not np.array_equal(tr_y, arr["y_train"]):
        raise ValueError("Train labels mismatch against fused target file.")
    if not np.array_equal(va_y, arr["y_val"]):
        raise ValueError("Val labels mismatch against fused target file.")
    if not np.array_equal(te_y, arr["y_test"]):
        raise ValueError("Test labels mismatch against fused target file.")

    if args.target_key == "probs_fused_bin":
        tr_t = arr["probs_fused_bin_train"]
        va_t = arr["probs_fused_bin_val"]
        te_t = arr["probs_fused_bin_test"]
    else:
        tr_t = arr["probs_fused_global_train"]
        va_t = arr["probs_fused_global_val"]
        te_t = arr["probs_fused_global_test"]

    ds_tr = DistillDataset(tr_feat, tr_mask, tr_y, tr_t)
    ds_va = DistillDataset(va_feat, va_mask, va_y, va_t)
    ds_te = DistillDataset(te_feat, te_mask, te_y, te_t)
    dl_tr = make_loader(ds_tr, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    dl_va = make_loader(ds_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    dl_te = make_loader(ds_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    model = JetClassTransformer(
        input_dim=int(tr_feat.shape[-1]),
        n_classes=int(len(class_names)),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sch = _scheduler(opt, warmup_epochs=int(args.warmup_epochs), total_epochs=int(args.epochs))

    best_val = float("inf")
    best_state = None
    wait = 0
    hist: List[Dict[str, float]] = []

    print("============================================================")
    print("Offline fused-student distillation")
    print("============================================================")
    print(f"Targets dir: {targets_dir}")
    print(f"Run ref dir: {run_ref_dir}")
    print(f"Save root:   {save_root}")
    print(f"Target key:  {args.target_key}")

    for ep in range(int(args.epochs)):
        model.train()
        tr_total = tr_kl = tr_ce = 0.0
        n_tr = 0
        n_cls = int(len(class_names))

        for batch in dl_tr:
            x = batch["feat"].to(device)
            m = batch["mask"].to(device)
            y = batch["label"].to(device)
            p_t = batch["target"].to(device)

            opt.zero_grad()
            logits = model(x, m)

            q_log = F.log_softmax(logits / float(args.distill_temp), dim=1)
            p_norm = torch.clamp(p_t, min=1e-8)
            p_norm = p_norm / p_norm.sum(dim=1, keepdim=True)
            kl_vec = torch.sum(p_norm * (torch.log(p_norm) - q_log), dim=1) * (float(args.distill_temp) ** 2)
            if bool(args.use_conf_weight):
                ent = -torch.sum(p_norm * torch.log(p_norm), dim=1)
                conf = 1.0 - ent / math.log(max(n_cls, 2))
                conf = torch.clamp(conf, min=0.05, max=1.0)
                loss_kl = (conf * kl_vec).mean()
            else:
                loss_kl = kl_vec.mean()
            loss_ce = F.cross_entropy(logits, y)
            loss = float(args.lambda_kl) * loss_kl + float(args.lambda_ce) * loss_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = int(y.shape[0])
            tr_total += float(loss.item()) * bs
            tr_kl += float(loss_kl.item()) * bs
            tr_ce += float(loss_ce.item()) * bs
            n_tr += bs

        sch.step()
        tr_total /= max(n_tr, 1)
        tr_kl /= max(n_tr, 1)
        tr_ce /= max(n_tr, 1)

        va = _eval_student(
            model=model,
            loader=dl_va,
            class_names=class_names,
            background_class=background_class,
            target_class=target_class,
            temp=float(args.distill_temp),
            lambda_kl=float(args.lambda_kl),
            lambda_ce=float(args.lambda_ce),
            use_conf_weight=bool(args.use_conf_weight),
        )
        va_loss = float(va["loss"])
        hist.append(
            {
                "epoch": int(ep + 1),
                "train_loss": float(tr_total),
                "train_kl": float(tr_kl),
                "train_ce": float(tr_ce),
                "val_loss": float(va_loss),
                "val_acc": float(va.get("acc", float("nan"))),
                "val_auc_macro": float(va.get("auc_macro_ovr", float("nan"))),
                "val_fpr50_sigbg": float(va.get("signal_vs_bg_fpr50", float("nan"))),
                "val_fpr50_ratio": float(va.get("target_vs_bg_ratio_fpr50", float("nan"))),
            }
        )

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 2 == 0:
            print(
                f"offline_fused_student ep {ep+1}: train(loss/kl/ce)="
                f"{tr_total:.4f}/{tr_kl:.4f}/{tr_ce:.4f} "
                f"val(loss/acc/auc/fpr50_sigbg/fpr50_ratio)="
                f"{va_loss:.4f}/{float(va.get('acc', float('nan'))):.4f}/"
                f"{float(va.get('auc_macro_ovr', float('nan'))):.4f}/"
                f"{float(va.get('signal_vs_bg_fpr50', float('nan'))):.6f}/"
                f"{float(va.get('target_vs_bg_ratio_fpr50', float('nan'))):.6f} "
                f"best={best_val:.4f}"
            )

        if wait >= int(args.patience):
            print(f"Early stopping offline fused student at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = _eval_student(
        model=model,
        loader=dl_te,
        class_names=class_names,
        background_class=background_class,
        target_class=target_class,
        temp=float(args.distill_temp),
        lambda_kl=float(args.lambda_kl),
        lambda_ce=float(args.lambda_ce),
        use_conf_weight=bool(args.use_conf_weight),
    )

    ckpt = {
        "model": model.state_dict(),
        "meta": {
            "input_dim": int(tr_feat.shape[-1]),
            "n_classes": int(len(class_names)),
            "embed_dim": int(args.embed_dim),
            "num_heads": int(args.num_heads),
            "num_layers": int(args.num_layers),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "class_names": list(class_names),
            "target_class": target_class,
            "background_class": background_class,
            "target_key": str(args.target_key),
        },
    }
    torch.save(ckpt, save_root / "offline_fused_student.pt")

    summary = {
        "targets_dir": str(targets_dir),
        "run_ref_dir": str(run_ref_dir),
        "save_root": str(save_root),
        "target_key": str(args.target_key),
        "best_val_loss": float(best_val),
        "val_last": {k: v for k, v in hist[-1].items() if k != "epoch"} if hist else {},
        "test_metrics": {k: v for k, v in te.items() if k not in ("probs", "labels")},
    }
    (save_root / "distill_history.json").write_text(json.dumps(hist, indent=2))
    (save_root / "distill_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        save_root / "offline_fused_student_test_scores.npz",
        labels=np.asarray(te["labels"], dtype=np.int64),
        probs=np.asarray(te["probs"], dtype=np.float32),
    )

    print("============================================================")
    print("Offline fused-student done")
    print("============================================================")
    print(
        f"Test: acc={float(te.get('acc', float('nan'))):.4f}, auc_macro={float(te.get('auc_macro_ovr', float('nan'))):.4f}, "
        f"fpr50(sig-vs-bg)={float(te.get('signal_vs_bg_fpr50', float('nan'))):.6f}, "
        f"fpr50({target_class}/({target_class}+{background_class}))="
        f"{float(te.get('target_vs_bg_ratio_fpr50', float('nan'))):.6f}"
    )
    print(f"Saved checkpoint: {save_root / 'offline_fused_student.pt'}")


if __name__ == "__main__":
    main()

