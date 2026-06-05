#!/usr/bin/env python3
"""
Train an HLT-only JetClass seed ensemble on one fixed split/HLT view.

Purpose:
  control whether stacked logistic regression over multiple independently
  trained HLT baselines can explain the large gains seen from reconstructor
  model fusion.

Important separation:
  - data_seed controls file split, event sampling, and HLT corruption;
  - train_seeds control model initialization and training order only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch

from evaluate_jetclass_hlt_teacher_baseline import (
    CANONICAL_CLASS_ORDER,
    HLTParams,
    JetDataset,
    collect_files_by_class,
    compute_features,
    eval_epoch,
    fit_model,
    get_mean_std,
    load_split,
    make_loader,
    split_files_by_class,
    standardize,
    summarize_hlt_diagnostics,
    set_seed,
)
from train_jetclass_joint_dualview_stage2_unmergeonly_m2hlt import _build_hlt_view_m2style


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    p.add_argument("--save_dir", type=Path, default=Path("checkpoints/jetclass_hlt_seed_ensemble"))
    p.add_argument("--run_prefix", type=str, default="hlt5_1m250k1m_fixedhlt_seed")
    p.add_argument("--data_seed", type=int, default=52)
    p.add_argument("--train_seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--feature_mode", type=str, default="full", choices=["kin", "kinpid", "full"])
    p.add_argument("--feature_preprocessing", type=str, default="canonical", choices=["canonical", "legacy"])
    p.add_argument("--class_assignment", type=str, default="filename", choices=["filename", "canonical_labels"])
    p.add_argument("--max_constits", type=int, default=128)
    p.add_argument("--train_files_per_class", type=int, default=8)
    p.add_argument("--val_files_per_class", type=int, default=1)
    p.add_argument("--test_files_per_class", type=int, default=1)
    p.add_argument("--shuffle_files", action="store_true", default=False)
    p.add_argument("--n_train_jets", type=int, default=1000000)
    p.add_argument("--n_val_jets", type=int, default=250000)
    p.add_argument("--n_test_jets", type=int, default=1000000)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Fixed m2-style HLT corruption profile.
    p.add_argument("--hlt_pt_threshold", type=float, default=1.30)
    p.add_argument("--merge_prob_scale", type=float, default=1.35)
    p.add_argument("--reassign_scale", type=float, default=1.00)
    p.add_argument("--smear_scale", type=float, default=1.00)
    p.add_argument("--eff_plateau_barrel", type=float, default=0.99)
    p.add_argument("--eff_plateau_endcap", type=float, default=0.97)
    p.add_argument("--eff_turnon_pt", type=float, default=1.40)
    p.add_argument("--eff_width_pt", type=float, default=0.20)
    p.add_argument("--target_class", type=str, default="Hbb")
    p.add_argument("--background_class", type=str, default="QCD")
    return p.parse_args()


def _run_args_for_child(args: argparse.Namespace, train_seed: int, run_name: str) -> SimpleNamespace:
    child = SimpleNamespace(**vars(args))
    child.seed = int(args.data_seed)
    child.training_seed = int(train_seed)
    child.run_name = str(run_name)
    child.save_dir = Path(args.save_dir)
    child.hlt_builder = "m2"
    return child


def _save_child_args(save_dir: Path, child_args: SimpleNamespace) -> None:
    payload = vars(child_args).copy()
    payload["save_dir"] = str(payload["save_dir"])
    payload["data_dir"] = str(payload["data_dir"])
    with (save_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _class_names(args: argparse.Namespace, files_by_class: Dict[str, Sequence[Path]]) -> List[str]:
    if str(args.class_assignment) == "canonical_labels":
        return list(CANONICAL_CLASS_ORDER)
    return sorted(files_by_class.keys())


def _load_fixed_data(args: argparse.Namespace):
    files_by_class = collect_files_by_class(args.data_dir.resolve())
    class_names = _class_names(args, files_by_class)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    if str(args.class_assignment) == "filename":
        source_files = {c: files_by_class[c] for c in class_names}
    else:
        source_files = files_by_class

    tr_files, va_files, te_files = split_files_by_class(
        source_files,
        n_train=int(args.train_files_per_class),
        n_val=int(args.val_files_per_class),
        n_test=int(args.test_files_per_class),
        shuffle=bool(args.shuffle_files),
        seed=int(args.data_seed),
    )

    print("Loading train split...")
    tr_tok, tr_mask, tr_y = load_split(
        tr_files,
        n_total=int(args.n_train_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.data_seed) + 101,
        class_assignment=str(args.class_assignment),
    )
    print("Loading val split...")
    va_tok, va_mask, va_y = load_split(
        va_files,
        n_total=int(args.n_val_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.data_seed) + 202,
        class_assignment=str(args.class_assignment),
    )
    print("Loading test split...")
    te_tok, te_mask, te_y = load_split(
        te_files,
        n_total=int(args.n_test_jets),
        max_constits=int(args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(args.data_seed) + 303,
        class_assignment=str(args.class_assignment),
    )

    hlt_params = HLTParams(
        hlt_pt_threshold=float(args.hlt_pt_threshold),
        merge_prob_scale=float(args.merge_prob_scale),
        reassign_scale=float(args.reassign_scale),
        smear_scale=float(args.smear_scale),
        eff_plateau_barrel=float(args.eff_plateau_barrel),
        eff_plateau_endcap=float(args.eff_plateau_endcap),
        eff_turnon_pt=float(args.eff_turnon_pt),
        eff_width_pt=float(args.eff_width_pt),
    )
    print("Building fixed m2-style HLT splits...")
    tr_hlt_tok, tr_hlt_mask, tr_hlt_diag = _build_hlt_view_m2style(
        tr_tok, tr_mask, params=hlt_params, seed=int(args.data_seed) + 1001
    )
    va_hlt_tok, va_hlt_mask, va_hlt_diag = _build_hlt_view_m2style(
        va_tok, va_mask, params=hlt_params, seed=int(args.data_seed) + 1002
    )
    te_hlt_tok, te_hlt_mask, te_hlt_diag = _build_hlt_view_m2style(
        te_tok, te_mask, params=hlt_params, seed=int(args.data_seed) + 1003
    )

    tr_feat_hlt = compute_features(
        tr_hlt_tok,
        tr_hlt_mask,
        feature_mode=str(args.feature_mode),
        feature_preprocessing=str(args.feature_preprocessing),
    )
    va_feat_hlt = compute_features(
        va_hlt_tok,
        va_hlt_mask,
        feature_mode=str(args.feature_mode),
        feature_preprocessing=str(args.feature_preprocessing),
    )
    te_feat_hlt = compute_features(
        te_hlt_tok,
        te_hlt_mask,
        feature_mode=str(args.feature_mode),
        feature_preprocessing=str(args.feature_preprocessing),
    )

    if str(args.feature_preprocessing) == "canonical":
        standardization_mode = "canonical_manual_fixed"
    else:
        tr_feat_off = compute_features(
            tr_tok,
            tr_mask,
            feature_mode=str(args.feature_mode),
            feature_preprocessing=str(args.feature_preprocessing),
        )
        mean, std = get_mean_std(tr_feat_off, tr_mask, np.arange(len(tr_y)))
        tr_feat_hlt = standardize(tr_feat_hlt, tr_hlt_mask, mean, std)
        va_feat_hlt = standardize(va_feat_hlt, va_hlt_mask, mean, std)
        te_feat_hlt = standardize(te_feat_hlt, te_hlt_mask, mean, std)
        standardization_mode = "learned_train_offline_split"

    ds_tr = JetDataset(tr_feat_hlt, tr_hlt_mask, tr_y)
    ds_va = JetDataset(va_feat_hlt, va_hlt_mask, va_y)
    ds_te = JetDataset(te_feat_hlt, te_hlt_mask, te_y)
    return {
        "class_names": class_names,
        "standardization_mode": standardization_mode,
        "datasets": (ds_tr, ds_va, ds_te),
        "masks": (tr_mask, va_mask, te_mask, tr_hlt_mask, va_hlt_mask, te_hlt_mask),
        "diagnostics": {
            "train": summarize_hlt_diagnostics(tr_hlt_diag),
            "val": summarize_hlt_diagnostics(va_hlt_diag),
            "test": summarize_hlt_diagnostics(te_hlt_diag),
        },
        "split_sizes": {"train": int(len(tr_y)), "val": int(len(va_y)), "test": int(len(te_y))},
        "input_dim": int(tr_feat_hlt.shape[-1]),
    }


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.data_seed))
    data = _load_fixed_data(args)
    ds_tr, ds_va, ds_te = data["datasets"]
    class_names = list(data["class_names"])
    n_classes = len(class_names)
    input_dim = int(data["input_dim"])

    ensemble_summary = {
        "purpose": "HLT-only seed ensemble control for stacked-logistic fusion",
        "data_seed": int(args.data_seed),
        "train_seeds": [int(s) for s in args.train_seeds],
        "split_sizes": data["split_sizes"],
        "class_names": class_names,
        "feature_mode": str(args.feature_mode),
        "feature_preprocessing": str(args.feature_preprocessing),
        "feature_standardization_mode": str(data["standardization_mode"]),
        "class_assignment": str(args.class_assignment),
        "hlt_builder": "m2",
        "hlt_params": {
            "hlt_pt_threshold": float(args.hlt_pt_threshold),
            "merge_prob_scale": float(args.merge_prob_scale),
            "reassign_scale": float(args.reassign_scale),
            "smear_scale": float(args.smear_scale),
            "eff_plateau_barrel": float(args.eff_plateau_barrel),
            "eff_plateau_endcap": float(args.eff_plateau_endcap),
            "eff_turnon_pt": float(args.eff_turnon_pt),
            "eff_width_pt": float(args.eff_width_pt),
        },
        "hlt_diagnostics": data["diagnostics"],
        "runs": [],
    }

    for idx, train_seed in enumerate(args.train_seeds, start=1):
        run_name = f"{args.run_prefix}{idx:02d}_trainseed{int(train_seed)}"
        child_save_dir = (args.save_dir / run_name).resolve()
        child_save_dir.mkdir(parents=True, exist_ok=True)
        child_args = _run_args_for_child(args, int(train_seed), run_name)
        _save_child_args(child_save_dir, child_args)

        print("\n" + "=" * 70)
        print(f"Training HLT baseline {idx}/{len(args.train_seeds)}: {run_name}")
        print(f"data_seed={args.data_seed} train_seed={train_seed}")
        print("=" * 70)
        set_seed(int(train_seed))
        dl_tr = make_loader(ds_tr, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
        dl_va = make_loader(ds_va, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
        dl_te = make_loader(ds_te, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
        model, best_val, history = fit_model(
            train_loader=dl_tr,
            val_loader=dl_va,
            input_dim=input_dim,
            n_classes=n_classes,
            class_names=class_names,
            background_class=str(args.background_class),
            target_class=str(args.target_class),
            args=child_args,
            tag="baseline_hlt",
            save_dir=child_save_dir,
        )
        device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
        model = model.to(device)
        test_metrics = eval_epoch(
            model,
            dl_te,
            device=device,
            class_names=class_names,
            background_class=str(args.background_class),
            target_class=str(args.target_class),
        )
        run_summary = {
            "run_name": run_name,
            "run_dir": str(child_save_dir),
            "data_seed": int(args.data_seed),
            "training_seed": int(train_seed),
            "baseline_val_best": best_val,
            "baseline_test": test_metrics,
            "checkpoint": str(child_save_dir / "baseline_hlt_best.pt"),
        }
        with (child_save_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **ensemble_summary,
                    "runs": [run_summary],
                    "baseline_val_best": best_val,
                    "baseline_test": test_metrics,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        with (child_save_dir / "baseline_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        ensemble_summary["runs"].append(run_summary)
        print(
            f"Finished {run_name}: test acc={test_metrics['acc']:.6f} "
            f"auc={test_metrics['auc_macro_ovr']:.6f} "
            f"fpr50={test_metrics['signal_vs_bg_fpr50']:.6f}"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (args.save_dir / f"{args.run_prefix}_ensemble_summary.json").open("w", encoding="utf-8") as f:
        json.dump(ensemble_summary, f, indent=2, sort_keys=True)
    print("\nSaved ensemble summary:")
    print(args.save_dir / f"{args.run_prefix}_ensemble_summary.json")


if __name__ == "__main__":
    main()
