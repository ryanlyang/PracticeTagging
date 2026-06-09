#!/usr/bin/env python3
"""Audit whether supervised logit decoding also boosts the offline teacher.

This answers a specific sanity question:

    If a logistic-regression decoder can strongly improve an HLT model's
    logits, does the same happen for the offline teacher logits?

The script regenerates logits from saved checkpoints on a val/test split,
then compares:

  - raw argmax accuracy
  - top-k accuracy
  - restricted diagonal one-vs-rest calibration
  - full multinomial logistic regression over logits/probs
  - permuted-label and row-shuffled controls

It intentionally does not use any reconstructor outputs or fusion scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from evaluate_jetclass_hlt_teacher_baseline import (
    CANONICAL_CLASS_ORDER,
    CLASS_NAME_ALIASES,
    HLTParams,
    JetClassTransformer,
    build_hlt_view,
    collect_files_by_class,
    compute_features,
    get_mean_std,
    load_split,
    split_files_by_class,
    standardize,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def _topk_acc(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argpartition(probs, kth=probs.shape[1] - k, axis=1)[:, -k:]
    return float(np.mean([int(y[i]) in set(order[i].tolist()) for i in range(len(y))]))


def _macro_auc(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        y_1h = np.eye(probs.shape[1], dtype=np.int64)[y]
        return float(roc_auc_score(y_1h, probs, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def _subset_indices(n: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_rows), replace=False))


def _fit_full_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    cs: Sequence[float],
    cv: int,
    max_iter: int,
    n_jobs: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegressionCV(
            Cs=[float(c) for c in cs],
            cv=int(cv),
            solver="lbfgs",
            scoring="accuracy",
            max_iter=int(max_iter),
            n_jobs=int(n_jobs),
            refit=True,
        ),
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_eval).astype(np.float32)
    lr = clf.named_steps["logisticregressioncv"]
    return probs, {
        "C_mean": float(np.mean(lr.C_)) if getattr(lr, "C_", np.array([])).size else float("nan"),
        "n_features": float(x_train.shape[1]),
    }


def _fit_diag_ovr(
    logits_train: np.ndarray,
    probs_train: np.ndarray,
    y_train: np.ndarray,
    logits_eval: np.ndarray,
    probs_eval: np.ndarray,
    *,
    max_iter: int,
) -> np.ndarray:
    n_classes = int(probs_train.shape[1])
    scores = np.zeros((logits_eval.shape[0], n_classes), dtype=np.float32)
    for c in range(n_classes):
        x_tr = np.stack([logits_train[:, c], probs_train[:, c]], axis=1).astype(np.float32)
        x_ev = np.stack([logits_eval[:, c], probs_eval[:, c]], axis=1).astype(np.float32)
        y_bin = (y_train == c).astype(np.int64)
        if len(np.unique(y_bin)) < 2:
            scores[:, c] = -1e6
            continue
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=int(max_iter)),
        )
        clf.fit(x_tr, y_bin)
        scores[:, c] = clf.decision_function(x_ev).astype(np.float32)
    return _softmax(scores)


class _ArrayDataset(Dataset):
    def __init__(self, feat: np.ndarray, mask: np.ndarray, label: np.ndarray):
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, i: int):
        return self.feat[i], self.mask[i], self.label[i]


def _resolve_ckpt(run_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these checkpoint files exist in {run_dir}: {list(candidates)}")


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "teacher", "baseline"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError(f"Could not identify a state_dict in {path}")


def _make_model(run_args: Dict[str, object], input_dim: int, n_classes: int) -> JetClassTransformer:
    return JetClassTransformer(
        input_dim=input_dim,
        n_classes=n_classes,
        embed_dim=int(run_args.get("embed_dim", 128)),
        num_heads=int(run_args.get("num_heads", 8)),
        num_layers=int(run_args.get("num_layers", 6)),
        ff_dim=int(run_args.get("ff_dim", 512)),
        dropout=float(run_args.get("dropout", 0.1)),
    )


@torch.no_grad()
def _predict_logits(
    model: JetClassTransformer,
    feat: np.ndarray,
    mask: np.ndarray,
    label: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = _ArrayDataset(feat, mask, label)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    model.eval().to(device)
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for xb, mb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        out = model(xb, mb)
        logits.append(out.detach().cpu().numpy().astype(np.float32))
        labels.append(yb.numpy().astype(np.int64))
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def _class_names(files_by_class: Dict[str, object], class_assignment: str) -> List[str]:
    if class_assignment == "canonical_labels":
        return list(CANONICAL_CLASS_ORDER)
    # Filename assignment must keep the actual ROOT filename class keys.  The
    # loader indexes split_files by these keys; aliasing them to canonical names
    # would make filename-style runs fail or silently use a different ordering.
    return sorted(files_by_class.keys())


def _build_splits(args: argparse.Namespace, run_args: Dict[str, object]):
    data_dir = Path(args.data_dir or run_args.get("data_dir", "")).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    files_by_class = collect_files_by_class(data_dir)
    class_assignment = str(run_args.get("class_assignment", "filename"))
    class_names = _class_names(files_by_class, class_assignment)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_files_per_class = int(run_args.get("train_files_per_class", 8))
    val_files_per_class = int(run_args.get("val_files_per_class", 1))
    test_files_per_class = int(run_args.get("test_files_per_class", 1))
    shuffle_files = bool(run_args.get("shuffle_files", False))
    seed = int(run_args.get("seed", args.seed))

    train_files, val_files, test_files = split_files_by_class(
        files_by_class,
        train_files_per_class,
        val_files_per_class,
        test_files_per_class,
        shuffle_files,
        seed,
    )
    return data_dir, class_names, class_to_idx, train_files, val_files, test_files


def _features_for_split(
    tok: np.ndarray,
    mask: np.ndarray,
    *,
    run_args: Dict[str, object],
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    feature_mode = str(run_args.get("feature_mode", "full"))
    feature_preprocessing = str(run_args.get("feature_preprocessing", "canonical"))
    feat = compute_features(tok, mask, feature_mode, feature_preprocessing)
    if mean is not None and std is not None:
        feat = standardize(feat, mask, mean, std)
    return feat


def _load_eval_arrays(args: argparse.Namespace, run_args: Dict[str, object]):
    data_dir, class_names, class_to_idx, train_files, val_files, test_files = _build_splits(args, run_args)
    max_constits = int(run_args.get("max_constits", 128))
    seed = int(run_args.get("seed", args.seed))
    class_assignment = str(run_args.get("class_assignment", "filename"))

    n_val = int(args.n_val_jets if args.n_val_jets > 0 else run_args.get("n_val_jets", 250000))
    n_test = int(args.n_test_jets if args.n_test_jets > 0 else run_args.get("n_test_jets", 1000000))

    print(f"Loading val split ({n_val} jets)...")
    val_tok, val_mask, y_val = load_split(
        val_files,
        n_val,
        max_constits,
        class_to_idx,
        seed + 2,
        class_assignment=class_assignment,
    )
    print(f"Loading test split ({n_test} jets)...")
    test_tok, test_mask, y_test = load_split(
        test_files,
        n_test,
        max_constits,
        class_to_idx,
        seed + 3,
        class_assignment=class_assignment,
    )

    mean = std = None
    if str(run_args.get("feature_preprocessing", "canonical")) == "legacy":
        n_stats = int(min(max(args.n_stats_jets, 1), int(run_args.get("n_train_jets", 100000))))
        print(f"Loading train-stat split ({n_stats} jets) for legacy standardization...")
        tr_tok, tr_mask, _ = load_split(
            train_files,
            n_stats,
            max_constits,
            class_to_idx,
            seed + 1,
            class_assignment=class_assignment,
        )
        tr_feat = compute_features(
            tr_tok,
            tr_mask,
            str(run_args.get("feature_mode", "full")),
            str(run_args.get("feature_preprocessing", "legacy")),
        )
        mean, std = get_mean_std(tr_feat, tr_mask, np.arange(len(tr_feat)))

    offline_val_feat = _features_for_split(val_tok, val_mask, run_args=run_args, mean=mean, std=std)
    offline_test_feat = _features_for_split(test_tok, test_mask, run_args=run_args, mean=mean, std=std)

    return {
        "data_dir": data_dir,
        "class_names": class_names,
        "offline_val_feat": offline_val_feat,
        "offline_test_feat": offline_test_feat,
        "val_tok": val_tok,
        "val_mask": val_mask,
        "test_tok": test_tok,
        "test_mask": test_mask,
        "y_val": y_val,
        "y_test": y_test,
        "mean": mean,
        "std": std,
    }


def _build_hlt_features(args: argparse.Namespace, run_args: Dict[str, object], arrays: Dict[str, object]):
    seed = int(run_args.get("seed", args.seed))
    params = HLTParams(
        hlt_pt_threshold=float(run_args.get("hlt_pt_threshold", 1.30)),
        merge_prob_scale=float(run_args.get("merge_prob_scale", 1.35)),
        reassign_scale=float(run_args.get("reassign_scale", 1.00)),
        smear_scale=float(run_args.get("smear_scale", 1.00)),
        eff_plateau_barrel=float(run_args.get("eff_plateau_barrel", 0.99)),
        eff_plateau_endcap=float(run_args.get("eff_plateau_endcap", 0.97)),
        eff_turnon_pt=float(run_args.get("eff_turnon_pt", 1.40)),
        eff_width_pt=float(run_args.get("eff_width_pt", 0.20)),
    )
    print("Building HLT-like val split...")
    hlt_val_tok, hlt_val_mask, _ = build_hlt_view(
        arrays["val_tok"],
        arrays["val_mask"],
        params,
        seed + 11,
        return_provenance=False,
    )
    print("Building HLT-like test split...")
    hlt_test_tok, hlt_test_mask, _ = build_hlt_view(
        arrays["test_tok"],
        arrays["test_mask"],
        params,
        seed + 12,
        return_provenance=False,
    )
    hlt_val_feat = _features_for_split(
        hlt_val_tok,
        hlt_val_mask,
        run_args=run_args,
        mean=arrays["mean"],
        std=arrays["std"],
    )
    hlt_test_feat = _features_for_split(
        hlt_test_tok,
        hlt_test_mask,
        run_args=run_args,
        mean=arrays["mean"],
        std=arrays["std"],
    )
    return hlt_val_feat, hlt_val_mask, hlt_test_feat, hlt_test_mask


def _audit_logits(
    label: str,
    logits_val: np.ndarray,
    y_val_full: np.ndarray,
    logits_test: np.ndarray,
    y_test_full: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    tr_idx = _subset_indices(len(y_val_full), int(args.max_stack_train_rows), int(args.seed) + 101)
    te_idx = _subset_indices(len(y_test_full), int(args.max_eval_rows), int(args.seed) + 102)
    y_val = y_val_full[tr_idx]
    y_test = y_test_full[te_idx]
    lv = logits_val[tr_idx].astype(np.float32)
    lt = logits_test[te_idx].astype(np.float32)
    pv = _softmax(lv)
    pt = _softmax(lt)

    raw_pred = pt.argmax(axis=1)
    raw_acc = float(accuracy_score(y_test, raw_pred))
    row: Dict[str, object] = {
        "model": label,
        "n_stack_train": int(len(y_val)),
        "n_eval": int(len(y_test)),
        "raw_acc": raw_acc,
        "raw_auc": _macro_auc(y_test, pt),
        "raw_nll": float(log_loss(y_test, pt, labels=np.arange(pt.shape[1]))),
        "top2_acc": _topk_acc(pt, y_test, 2),
        "top3_acc": _topk_acc(pt, y_test, 3),
    }

    diag_probs = _fit_diag_ovr(lv, pv, y_val, lt, pt, max_iter=int(args.diag_max_iter))
    row["diag_ovr_acc"] = float(accuracy_score(y_test, diag_probs.argmax(axis=1)))
    row["diag_ovr_auc"] = _macro_auc(y_test, diag_probs)
    row["diag_ovr_gain"] = float(row["diag_ovr_acc"] - raw_acc)

    controls: List[Dict[str, object]] = []
    for mode, xv, xt in (
        ("logits", lv, lt),
        ("probs", pv, pt),
        ("logits_probs", np.concatenate([lv, pv], axis=1), np.concatenate([lt, pt], axis=1)),
    ):
        stack_probs, info = _fit_full_logreg(
            xv,
            y_val,
            xt,
            cs=args.Cs,
            cv=int(args.cv),
            max_iter=int(args.max_iter),
            n_jobs=int(args.n_jobs),
        )
        acc = float(accuracy_score(y_test, stack_probs.argmax(axis=1)))
        row[f"stack_{mode}_acc"] = acc
        row[f"stack_{mode}_auc"] = _macro_auc(y_test, stack_probs)
        row[f"stack_{mode}_gain"] = acc - raw_acc
        row[f"stack_{mode}_C_mean"] = info["C_mean"]

    # Controls on the most flexible input.
    xv = np.concatenate([lv, pv], axis=1)
    xt = np.concatenate([lt, pt], axis=1)
    rng = np.random.default_rng(int(args.seed) + 1000)
    y_perm = y_val.copy()
    rng.shuffle(y_perm)
    p_perm, _ = _fit_full_logreg(
        xv,
        y_perm,
        xt,
        cs=args.Cs,
        cv=int(args.cv),
        max_iter=int(args.max_iter),
        n_jobs=int(args.n_jobs),
    )
    controls.append({
        "model": label,
        "control": "permuted_stack_labels",
        "acc": float(accuracy_score(y_test, p_perm.argmax(axis=1))),
        "auc": _macro_auc(y_test, p_perm),
    })

    xv_shuf = xv.copy()
    for c in range(xv_shuf.shape[1]):
        rng.shuffle(xv_shuf[:, c])
    p_row, _ = _fit_full_logreg(
        xv_shuf,
        y_val,
        xt,
        cs=args.Cs,
        cv=int(args.cv),
        max_iter=int(args.max_iter),
        n_jobs=int(args.n_jobs),
    )
    controls.append({
        "model": label,
        "control": "row_shuffled_stack_features",
        "acc": float(accuracy_score(y_test, p_row.argmax(axis=1))),
        "auc": _macro_auc(y_test, p_row),
    })

    return row, controls


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--n_val_jets", type=int, default=250000)
    ap.add_argument("--n_test_jets", type=int, default=300000)
    ap.add_argument("--n_stats_jets", type=int, default=100000)
    ap.add_argument("--max_stack_train_rows", type=int, default=150000)
    ap.add_argument("--max_eval_rows", type=int, default=300000)
    ap.add_argument("--include_hlt_baseline", action="store_true")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--diag_max_iter", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json in {run_dir}")
    run_args = json.loads(args_path.read_text())

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("JetClass Offline Teacher Decoder Audit")
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {args.out_dir.resolve()}")
    print(f"Device:  {device}")
    print(f"n_val/n_test loaded: {args.n_val_jets}/{args.n_test_jets}")
    print(f"stack train/eval rows: {args.max_stack_train_rows}/{args.max_eval_rows}")
    print(f"include HLT baseline: {args.include_hlt_baseline}")
    print("=" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    arrays = _load_eval_arrays(args, run_args)
    class_names = arrays["class_names"]
    n_classes = len(class_names)

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []

    input_dim = int(arrays["offline_val_feat"].shape[-1])
    teacher_ckpt = _resolve_ckpt(run_dir, ["teacher_offline_best.pt", "teacher.pt"])
    print(f"Loading offline teacher: {teacher_ckpt}")
    teacher = _make_model(run_args, input_dim, n_classes)
    teacher.load_state_dict(_load_state_dict(teacher_ckpt), strict=True)
    logits_val, y_val = _predict_logits(
        teacher,
        arrays["offline_val_feat"],
        arrays["val_mask"],
        arrays["y_val"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    logits_test, y_test = _predict_logits(
        teacher,
        arrays["offline_test_feat"],
        arrays["test_mask"],
        arrays["y_test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    row, control = _audit_logits("offline_teacher", logits_val, y_val, logits_test, y_test, args)
    rows.append(row)
    controls.extend(control)

    if args.include_hlt_baseline:
        hlt_val_feat, hlt_val_mask, hlt_test_feat, hlt_test_mask = _build_hlt_features(args, run_args, arrays)
        baseline_ckpt = _resolve_ckpt(run_dir, ["baseline_hlt_best.pt", "baseline.pt"])
        print(f"Loading HLT baseline: {baseline_ckpt}")
        baseline = _make_model(run_args, int(hlt_val_feat.shape[-1]), n_classes)
        baseline.load_state_dict(_load_state_dict(baseline_ckpt), strict=True)
        hlt_logits_val, hlt_y_val = _predict_logits(
            baseline,
            hlt_val_feat,
            hlt_val_mask,
            arrays["y_val"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        hlt_logits_test, hlt_y_test = _predict_logits(
            baseline,
            hlt_test_feat,
            hlt_test_mask,
            arrays["y_test"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        row, control = _audit_logits("hlt_baseline_same_run", hlt_logits_val, hlt_y_val, hlt_logits_test, hlt_y_test, args)
        rows.append(row)
        controls.extend(control)

    result = {
        "run_dir": str(run_dir),
        "class_names": list(class_names),
        "args": vars(args),
        "metrics": rows,
        "controls": controls,
    }
    (args.out_dir / "offline_teacher_decoder_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_csv(args.out_dir / "offline_teacher_decoder_audit_metrics.csv", rows)
    _write_csv(args.out_dir / "offline_teacher_decoder_audit_controls.csv", controls)

    print("-" * 60)
    for r in rows:
        print(
            f"{r['model']:22s} raw={r['raw_acc']:.6f} "
            f"diag={r['diag_ovr_acc']:.6f} "
            f"stack_lp={r['stack_logits_probs_acc']:.6f} "
            f"gain={r['stack_logits_probs_gain']:+.6f} "
            f"top2={r['top2_acc']:.6f}"
        )
    for c in controls:
        print(f"control {c['model']:22s} {c['control']:28s} acc={c['acc']:.6f} auc={c['auc']:.6f}")
    print(f"Saved: {args.out_dir / 'offline_teacher_decoder_audit.json'}")


if __name__ == "__main__":
    main()
