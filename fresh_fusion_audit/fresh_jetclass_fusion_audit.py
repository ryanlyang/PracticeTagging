#!/usr/bin/env python3
"""Fresh JetClass fusion audit with explicit stack/train/val/test splits.

This script intentionally does not import the old stacked-fusion analyzer.
It reuses only low-level project components needed to load JetClass data,
build the fixed HLT view, and instantiate the already-trained models.

Protocol:
  1. Load one held-out source split once, deterministically and without
     overlapping event ranges within files.
  2. Slice that loaded sample into stack_train, stack_val, and final_test.
  3. Fit all trainable fusion pieces only on stack_train.
  4. Choose the best method by stack_val only.
  5. Report final_test metrics once for every method.

The main goal is to separate a legitimate model/fusion gain from a stacker
that is simply acting as a supervised second-stage classifier.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from evaluate_jetclass_hlt_teacher_baseline import (
    CANONICAL_CLASS_ORDER,
    HLTParams,
    JetClassTransformer,
    build_hlt_view,
    canonical_label_branches_for_classes,
    class_quota,
    collect_files_by_class,
    compute_features,
    eval_metrics,
    extract_chunk_class_indices,
    extract_tokens_from_chunk,
    get_first_tree,
    get_mean_std,
    load_split,
    required_raw_branches,
    resolve_branch_map,
    split_files_by_class,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import CONFIG as BASE_RECO_CONFIG
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_confgen_ops as confgen_ops
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_hybrid_ops as hybrid_ops
import train_jetclass_joint_dualview_stage2_unmergeonly_v2_attr as v2


@dataclass
class SourceSpec:
    name: str
    kind: str
    run_dir: Path


@dataclass
class LoadedSource:
    name: str
    kind: str
    run_dir: Path
    sources: Dict[str, str]
    predict_logits: Callable[[Dict[str, torch.Tensor]], torch.Tensor]


class FreshEvalDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt4: np.ndarray,
        feat_offline: np.ndarray,
        mask_offline: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt4 = torch.tensor(const_hlt4, dtype=torch.float32)
        self.feat_offline = torch.tensor(feat_offline, dtype=torch.float32)
        self.mask_offline = torch.tensor(mask_offline, dtype=torch.bool)
        self.labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt4": self.const_hlt4[i],
            "feat_offline": self.feat_offline[i],
            "mask_offline": self.mask_offline[i],
            "label": self.labels[i],
        }


def _parse_source(s: str) -> SourceSpec:
    parts = s.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --source `{s}`. Expected name:kind:run_dir")
    name, kind, run_dir = parts[0].strip(), parts[1].strip(), Path(parts[2].strip()).resolve()
    valid = {"baseline_hlt", "offline_teacher", "stage2", "joint", "reco_only_stagea"}
    if kind not in valid:
        raise ValueError(f"Unsupported source kind `{kind}`. Valid: {sorted(valid)}")
    return SourceSpec(name=name, kind=kind, run_dir=run_dir)


def _parse_stack_group(s: str) -> Tuple[str, List[str]]:
    if ":" not in s:
        raise ValueError(f"Invalid --stack_group `{s}`. Expected group_name:name1,name2,...")
    name, rest = s.split(":", 1)
    names = [x.strip() for x in rest.split(",") if x.strip()]
    if not name.strip() or not names:
        raise ValueError(f"Invalid --stack_group `{s}`.")
    return name.strip(), names


def _ns(path: Path) -> SimpleNamespace:
    return SimpleNamespace(**json.loads(path.read_text()))


def _get(ns: SimpleNamespace, key: str, default):
    return getattr(ns, key, default)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def _macro_auc(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        y_1h = np.eye(probs.shape[1], dtype=np.int64)[y]
        return float(roc_auc_score(y_1h, probs, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def _nll(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(log_loss(y, probs, labels=np.arange(probs.shape[1])))
    except Exception:
        idx = np.arange(len(y))
        return float(-np.mean(np.log(np.clip(probs[idx, y], 1e-12, 1.0))))


def _metrics(y: np.ndarray, probs: np.ndarray, class_names: Sequence[str], bg_name: str, tgt_name: str) -> Dict[str, float]:
    out = {}
    try:
        out.update(eval_metrics(y, probs, class_names, bg_name, tgt_name))
    except Exception:
        out["acc"] = float(accuracy_score(y, np.argmax(probs, axis=1)))
        out["auc_macro_ovr"] = _macro_auc(y, probs)
    out["acc"] = float(accuracy_score(y, np.argmax(probs, axis=1)))
    out["auc_macro_ovr"] = float(out.get("auc_macro_ovr", _macro_auc(y, probs)))
    out["nll"] = _nll(y, probs)
    return {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in out.items()}


def _load_state_dict(run_dir: Path, names: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], Path]:
    tried: List[str] = []
    for n in names:
        p = (run_dir / n).resolve()
        tried.append(str(p))
        if not p.exists():
            continue
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("model", "model_state_dict", "state_dict", "teacher", "baseline"):
                if key in obj and isinstance(obj[key], dict):
                    return obj[key], p
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj, p
        raise RuntimeError(f"Could not read state_dict from {p}")
    raise FileNotFoundError(f"No checkpoint found in {run_dir}. Tried: {tried}")


def _check_compat(ref: SimpleNamespace, other: SimpleNamespace, label: str) -> None:
    strict = (
        "feature_mode",
        "feature_preprocessing",
        "class_assignment",
        "target_class",
        "background_class",
        "max_constits",
        "train_files_per_class",
        "val_files_per_class",
        "test_files_per_class",
        "shuffle_files",
    )
    for k in strict:
        if str(getattr(ref, k, None)) != str(getattr(other, k, None)):
            raise ValueError(f"Incompatible run `{label}` for {k}: ref={getattr(ref, k, None)} other={getattr(other, k, None)}")

    hlt = (
        "hlt_pt_threshold",
        "merge_prob_scale",
        "reassign_scale",
        "smear_scale",
        "eff_plateau_barrel",
        "eff_plateau_endcap",
        "eff_turnon_pt",
        "eff_width_pt",
    )
    mismatches = []
    for k in hlt:
        if str(getattr(ref, k, None)) != str(getattr(other, k, None)):
            mismatches.append(f"{k}: ref={getattr(ref, k, None)} other={getattr(other, k, None)}")
    if mismatches:
        raise ValueError(f"HLT profile mismatch for `{label}`: " + "; ".join(mismatches))


def _read_seq_tokens_from_file(file_path: Path, max_constits: int, start: int, stop: int):
    tree = get_first_tree(file_path)
    bmap = resolve_branch_map(tree)
    branches = required_raw_branches(bmap)
    arr = tree.arrays(branches, entry_start=int(start), entry_stop=int(stop), library="ak")
    tok, mask = extract_tokens_from_chunk(arr, bmap, max_constits=max_constits)
    valid = mask.any(axis=1)
    return tok[valid], mask[valid]


def _load_disjoint_filename(
    split_files: Dict[str, List[Path]],
    n_total: int,
    max_constits: int,
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = sorted(class_to_idx.keys(), key=lambda c: int(class_to_idx[c]))
    quotas = class_quota(int(n_total), classes)
    all_tok: List[np.ndarray] = []
    all_mask: List[np.ndarray] = []
    all_lab: List[np.ndarray] = []
    chunk = 20000
    for cls in classes:
        need = int(quotas[cls])
        got_tok: List[np.ndarray] = []
        got_mask: List[np.ndarray] = []
        got = 0
        for file_path in split_files[cls]:
            tree = get_first_tree(file_path)
            n_entries = int(tree.num_entries)
            start = 0
            while got < need and start < n_entries:
                stop = min(start + chunk, n_entries)
                tok, mask = _read_seq_tokens_from_file(file_path, max_constits, start, stop)
                if len(tok):
                    take = min(need - got, len(tok))
                    got_tok.append(tok[:take])
                    got_mask.append(mask[:take])
                    got += take
                start = stop
            if got >= need:
                break
        if got < need:
            raise RuntimeError(f"Could not load enough disjoint filename events for {cls}: got {got}, need {need}")
        tok_c = np.concatenate(got_tok, axis=0)[:need]
        mask_c = np.concatenate(got_mask, axis=0)[:need]
        lab_c = np.full((len(tok_c),), int(class_to_idx[cls]), dtype=np.int64)
        all_tok.append(tok_c)
        all_mask.append(mask_c)
        all_lab.append(lab_c)
    return np.concatenate(all_tok), np.concatenate(all_mask), np.concatenate(all_lab)


def _load_disjoint_canonical(
    split_files: Dict[str, List[Path]],
    n_total: int,
    max_constits: int,
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = sorted(class_to_idx.keys(), key=lambda c: int(class_to_idx[c]))
    quotas = class_quota(int(n_total), classes)
    label_branches = canonical_label_branches_for_classes(classes)
    got_tok: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    got_mask: Dict[str, List[np.ndarray]] = {c: [] for c in classes}
    counts = {c: 0 for c in classes}
    shared = sorted({p for paths in split_files.values() for p in paths})
    chunk = 20000
    for file_path in shared:
        tree = get_first_tree(file_path)
        bmap = resolve_branch_map(tree)
        branches = sorted(set(required_raw_branches(bmap) + label_branches))
        n_entries = int(tree.num_entries)
        start = 0
        while start < n_entries and any(counts[c] < quotas[c] for c in classes):
            stop = min(start + chunk, n_entries)
            arr = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")
            tok, mask = extract_tokens_from_chunk(arr, bmap, max_constits=max_constits)
            valid = mask.any(axis=1)
            if valid.any():
                tok_v = tok[valid]
                mask_v = mask[valid]
                y_idx = extract_chunk_class_indices(arr, label_branches)[valid]
                for cls in classes:
                    need = int(quotas[cls]) - int(counts[cls])
                    if need <= 0:
                        continue
                    cls_idx = int(class_to_idx[cls])
                    sel = np.flatnonzero(y_idx == cls_idx)
                    if sel.size == 0:
                        continue
                    take = sel[:need]
                    got_tok[cls].append(tok_v[take])
                    got_mask[cls].append(mask_v[take])
                    counts[cls] += int(take.size)
            start = stop
    missing = [f"{c}:{counts[c]}/{quotas[c]}" for c in classes if counts[c] < quotas[c]]
    if missing:
        raise RuntimeError("Could not satisfy canonical quotas: " + ", ".join(missing))
    all_tok: List[np.ndarray] = []
    all_mask: List[np.ndarray] = []
    all_lab: List[np.ndarray] = []
    for cls in classes:
        need = int(quotas[cls])
        tok_c = np.concatenate(got_tok[cls], axis=0)[:need]
        mask_c = np.concatenate(got_mask[cls], axis=0)[:need]
        lab_c = np.full((len(tok_c),), int(class_to_idx[cls]), dtype=np.int64)
        all_tok.append(tok_c)
        all_mask.append(mask_c)
        all_lab.append(lab_c)
    return np.concatenate(all_tok), np.concatenate(all_mask), np.concatenate(all_lab)


def _build_source_split_files(ref: SimpleNamespace, data_dir: Path):
    files_by_class = collect_files_by_class(data_dir.resolve())
    if str(ref.class_assignment) == "canonical_labels":
        class_names = list(CANONICAL_CLASS_ORDER)
    else:
        class_names = sorted(files_by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    tr_files, va_files, te_files = split_files_by_class(
        files_by_class,
        n_train=int(ref.train_files_per_class),
        n_val=int(ref.val_files_per_class),
        n_test=int(ref.test_files_per_class),
        shuffle=bool(ref.shuffle_files),
        seed=int(ref.seed),
    )
    return class_names, class_to_idx, {"train": tr_files, "val": va_files, "test": te_files}


def _load_fresh_source_events(
    ref: SimpleNamespace,
    data_dir: Path,
    source_split: str,
    total_events: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    class_names, class_to_idx, splits = _build_source_split_files(ref, data_dir)
    split_files = splits[str(source_split)]
    print(f"Loading fresh source events from original `{source_split}` files: n={total_events}")
    if str(ref.class_assignment) == "canonical_labels":
        tok, mask, y = _load_disjoint_canonical(split_files, total_events, int(ref.max_constits), class_to_idx)
    else:
        tok, mask, y = _load_disjoint_filename(split_files, total_events, int(ref.max_constits), class_to_idx)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(y))
    return class_names, tok[perm], mask[perm], y[perm]


def _feature_arrays(ref: SimpleNamespace, tok: np.ndarray, mask: np.ndarray, data_dir: Path):
    feat = compute_features(tok, mask, str(ref.feature_mode), str(ref.feature_preprocessing))
    if str(ref.feature_preprocessing) == "canonical":
        return feat.astype(np.float32), "canonical_manual_fixed"

    # Learned standardization only for legacy/non-canonical preprocessing.
    class_names, class_to_idx, splits = _build_source_split_files(ref, data_dir)
    tr_tok, tr_mask, _ = load_split(
        splits["train"],
        n_total=min(int(ref.n_train_jets), 100000),
        max_constits=int(ref.max_constits),
        class_to_idx=class_to_idx,
        seed=int(ref.seed) + 101,
        class_assignment=str(ref.class_assignment),
    )
    tr_feat = compute_features(tr_tok, tr_mask, str(ref.feature_mode), str(ref.feature_preprocessing))
    mean, std = get_mean_std(tr_feat, tr_mask, np.arange(len(tr_feat)))
    return standardize(feat, mask, mean, std), "learned_train_split"


def _split_indices(n_stack_train: int, n_stack_val: int, n_final_test: int) -> Dict[str, slice]:
    a = int(n_stack_train)
    b = a + int(n_stack_val)
    c = b + int(n_final_test)
    return {"stack_train": slice(0, a), "stack_val": slice(a, b), "final_test": slice(b, c)}


def _hash_rows(tok: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> List[str]:
    out: List[str] = []
    for i in idx:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(tok[i, :, :4]).tobytes())
        h.update(np.ascontiguousarray(mask[i]).tobytes())
        out.append(h.hexdigest())
    return out


def _hash_audit(tok: np.ndarray, mask: np.ndarray, splits: Dict[str, slice], rows_per_split: int, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(int(seed))
    sets: Dict[str, set] = {}
    counts: Dict[str, int] = {}
    for name, sl in splits.items():
        n = sl.stop - sl.start
        take = min(int(rows_per_split), n)
        local = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)
        idx = local + sl.start
        hs = set(_hash_rows(tok, mask, idx))
        sets[name] = hs
        counts[name] = len(hs)
    overlaps = {}
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlaps[f"{names[i]}__{names[j]}"] = int(len(sets[names[i]] & sets[names[j]]))
    return {"sampled_unique_hash_counts": counts, "sampled_overlap_counts": overlaps}


def _make_single_view(ref: SimpleNamespace, input_dim: int, n_classes: int, device: torch.device) -> JetClassTransformer:
    return JetClassTransformer(
        input_dim=int(input_dim),
        n_classes=int(n_classes),
        embed_dim=int(ref.embed_dim),
        num_heads=int(ref.num_heads),
        num_layers=int(ref.num_layers),
        ff_dim=int(ref.ff_dim),
        dropout=float(ref.dropout),
    ).to(device)


def _detect_reco_family(sd: Dict[str, torch.Tensor]) -> str:
    keys = tuple(sd.keys())
    if "base.gate_temperature" in sd or any(k.startswith("base.op_gate_head.") for k in keys):
        return "confgen_ops"
    if any(k.startswith("base.split_exist_head.") for k in keys) or any(k.startswith("base.gen_attn.") for k in keys):
        return "hybrid_ops"
    return "confgen_ops"


def _build_reco(ref: SimpleNamespace, input_dim: int, device: torch.device, family: str) -> torch.nn.Module:
    cfg = copy.deepcopy(BASE_RECO_CONFIG)
    defaults = cfg["reconstructor_model"]
    cfg["reconstructor_model"]["embed_dim"] = int(_get(ref, "reco_embed_dim", defaults["embed_dim"]))
    cfg["reconstructor_model"]["num_heads"] = int(_get(ref, "reco_num_heads", defaults["num_heads"]))
    cfg["reconstructor_model"]["num_layers"] = int(_get(ref, "reco_num_layers", defaults["num_layers"]))
    cfg["reconstructor_model"]["ff_dim"] = int(_get(ref, "reco_ff_dim", defaults["ff_dim"]))
    cfg["reconstructor_model"]["dropout"] = float(_get(ref, "reco_dropout", defaults["dropout"]))
    cfg["reconstructor_model"]["max_split_children"] = int(_get(ref, "reco_max_split_children", defaults["max_split_children"]))
    cfg["reconstructor_model"]["max_generated_tokens"] = int(_get(ref, "reco_max_generated_tokens", defaults["max_generated_tokens"]))
    if family == "hybrid_ops":
        cfg["reconstructor_model"]["edit_delta_scale"] = float(_get(ref, "specialist_edit_delta_scale", 1.0))
        cfg["reconstructor_model"]["split_weight_scale"] = float(_get(ref, "specialist_split_weight_scale", 1.0))
        cfg["reconstructor_model"]["gen_weight_scale"] = float(_get(ref, "specialist_gen_weight_scale", 1.0))
        base_reco = hybrid_ops.OfflineReconstructorHybridOps(input_dim=int(input_dim), **cfg["reconstructor_model"]).to(device)
    else:
        base_reco = confgen_ops.OfflineReconstructorConfidenceHybridOps(input_dim=int(input_dim), **cfg["reconstructor_model"]).to(device)
    return v2.ReconstructorWithAttrHeads(
        base=base_reco,
        input_dim=int(input_dim),
        hidden_dim=int(_get(ref, "v2_attr_hidden_dim", 128)),
        attr_slots=int(_get(ref, "v2_attr_slots", 2)),
    ).to(device)


def _build_dual(ref: SimpleNamespace, input_dim: int, n_classes: int, device: torch.device) -> torch.nn.Module:
    return v2.JetClassDualViewTransformer(
        input_dim_a=int(input_dim),
        input_dim_b=10,
        n_classes=int(n_classes),
        embed_dim=int(ref.embed_dim),
        num_heads=int(ref.num_heads),
        num_layers=int(ref.num_layers),
        ff_dim=int(ref.ff_dim),
        dropout=float(ref.dropout),
    ).to(device)


def _soft_corrected_view(family: str, reco_out: Dict[str, torch.Tensor], weight_floor: float):
    if family == "hybrid_ops":
        return hybrid_ops.build_soft_corrected_view_hybrid_ops(
            reco_out,
            weight_floor=float(weight_floor),
            scale_features_by_weight=True,
            include_flags=False,
        )
    return confgen_ops.build_soft_corrected_view_confgen_ops(
        reco_out,
        weight_floor=float(weight_floor),
        scale_features_by_weight=True,
        include_flags=False,
    )


def _load_source(
    spec: SourceSpec,
    ref: SimpleNamespace,
    input_dim: int,
    n_classes: int,
    device: torch.device,
    corrected_weight_floor: float,
) -> LoadedSource:
    run_dir = spec.run_dir
    if spec.kind == "baseline_hlt":
        model = _make_single_view(ref, input_dim, n_classes, device)
        sd, ckpt = _load_state_dict(run_dir, ("baseline_hlt_best.pt", "baseline_best.pt", "baseline.pt"))
        model.load_state_dict(sd, strict=True)
        model.eval()

        def predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(batch["feat_hlt"].to(device), batch["mask_hlt"].to(device))

        return LoadedSource(spec.name, spec.kind, run_dir, {"model_ckpt": str(ckpt)}, predict)

    if spec.kind == "offline_teacher":
        model = _make_single_view(ref, input_dim, n_classes, device)
        sd, ckpt = _load_state_dict(run_dir, ("teacher_offline_best.pt", "teacher.pt"))
        model.load_state_dict(sd, strict=True)
        model.eval()

        def predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return model(batch["feat_offline"].to(device), batch["mask_offline"].to(device))

        return LoadedSource(spec.name, spec.kind, run_dir, {"model_ckpt": str(ckpt)}, predict)

    if spec.kind in {"stage2", "joint", "reco_only_stagea"}:
        reco_sd, reco_ckpt = _load_state_dict(run_dir, ("offline_reconstructor_stage2.pt", "offline_reconstructor.pt"))
        family = _detect_reco_family(reco_sd)
        reco = _build_reco(ref, input_dim, device, family)
        try:
            reco.load_state_dict(reco_sd, strict=True)
        except RuntimeError as exc:
            print(f"[load-warning] strict reco load failed for {spec.name}; retrying strict=False. Error: {exc}")
            reco.load_state_dict(reco_sd, strict=False)
        reco.eval()

        if spec.kind == "reco_only_stagea":
            clf = _make_single_view(ref, 10, n_classes, device)
            clf_sd, clf_ckpt = _load_state_dict(run_dir, ("reco_only_corrected_stageA_best.pt", "reco_only_corrected_stageA.pt"))
            clf.load_state_dict(clf_sd, strict=True)
            clf.eval()

            def predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
                x = batch["feat_hlt"].to(device)
                m = batch["mask_hlt"].to(device)
                c4 = batch["const_hlt4"].to(device)
                reco_out = reco(x, m, c4, stage_scale=1.0)
                feat_b, mask_b = _soft_corrected_view(family, reco_out, corrected_weight_floor)
                return clf(feat_b, mask_b)

            return LoadedSource(
                spec.name,
                spec.kind,
                run_dir,
                {"reco_ckpt": str(reco_ckpt), "reco_family": family, "classifier_ckpt": str(clf_ckpt)},
                predict,
            )

        dual = _build_dual(ref, input_dim, n_classes, device)
        dual_names = ("dual_joint_stage2.pt", "dual_joint.pt") if spec.kind == "stage2" else ("dual_joint.pt", "dual_joint_stage2.pt")
        dual_sd, dual_ckpt = _load_state_dict(run_dir, dual_names)
        dual.load_state_dict(dual_sd, strict=True)
        dual.eval()

        def predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            x = batch["feat_hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            c4 = batch["const_hlt4"].to(device)
            reco_out = reco(x, m, c4, stage_scale=1.0)
            feat_b, mask_b = _soft_corrected_view(family, reco_out, corrected_weight_floor)
            return dual(x, m, feat_b, mask_b)

        return LoadedSource(
            spec.name,
            spec.kind,
            run_dir,
            {"reco_ckpt": str(reco_ckpt), "reco_family": family, "dual_ckpt": str(dual_ckpt)},
            predict,
        )

    raise ValueError(f"Unsupported kind {spec.kind}")


@torch.no_grad()
def _collect_logits(source: LoadedSource, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in loader:
        out = source.predict_logits(batch)
        logits.append(out.detach().cpu().numpy().astype(np.float32))
        labels.append(batch["label"].numpy().astype(np.int64))
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def _fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 80) -> Tuple[float, float, float]:
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y.astype(np.int64), dtype=torch.long)
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe")
    before = float(F.cross_entropy(lt, yt).item())

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3, max=1e3)
        loss = F.cross_entropy(lt / t, yt)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t.detach()).clamp(min=1e-3, max=1e3).item())
    after = float(F.cross_entropy(lt / t, yt).item())
    return t, before, after


def _fit_logreg_model(x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace):
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegressionCV(
            Cs=[float(c) for c in args.stack_Cs],
            cv=int(args.stack_cv),
            solver="lbfgs",
            scoring="accuracy",
            max_iter=int(args.stack_max_iter),
            n_jobs=int(args.stack_n_jobs),
            refit=True,
        ),
    )
    clf.fit(x_train, y_train)
    lr = clf.named_steps["logisticregressioncv"]
    info = {"C_mean": float(np.mean(lr.C_)) if getattr(lr, "C_", np.array([])).size else float("nan")}
    return clf, info


def _sample_weights(n_models: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    samples = rng.dirichlet(np.ones((n_models,), dtype=np.float64), size=int(max(1, n_samples)))
    return np.concatenate([np.full((1, n_models), 1.0 / n_models), np.eye(n_models), samples], axis=0).astype(np.float32)


def _fuse_probs(weights: np.ndarray, probs_list: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(probs_list[0], dtype=np.float64)
    for w, p in zip(weights, probs_list):
        out += float(w) * p.astype(np.float64)
    return out.astype(np.float32)


def _fuse_logits(weights: np.ndarray, logits_list: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(logits_list[0], dtype=np.float64)
    for w, z in zip(weights, logits_list):
        out += float(w) * z.astype(np.float64)
    return out.astype(np.float32)


def _search_weights(y: np.ndarray, arrs: Sequence[np.ndarray], mode: str, n_samples: int, seed: int) -> Tuple[np.ndarray, float]:
    weights = _sample_weights(len(arrs), n_samples, seed)
    best_w = weights[0]
    best_acc = -1.0
    for w in weights:
        if mode == "prob":
            p = _fuse_probs(w, arrs)
        else:
            p = _softmax(_fuse_logits(w, arrs))
        acc = float(np.mean(np.argmax(p, axis=1) == y))
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w.astype(np.float32), float(best_acc)


def _stack_features(logits_list: Sequence[np.ndarray], probs_list: Sequence[np.ndarray], mode: str) -> np.ndarray:
    if mode == "logits":
        return np.concatenate(logits_list, axis=1).astype(np.float32)
    if mode == "probs":
        return np.concatenate(probs_list, axis=1).astype(np.float32)
    return np.concatenate([np.concatenate(logits_list, axis=1), np.concatenate(probs_list, axis=1)], axis=1).astype(np.float32)


def _evaluate_group(
    group_name: str,
    source_names: Sequence[str],
    logits_by_source: Dict[str, np.ndarray],
    y: np.ndarray,
    splits: Dict[str, slice],
    class_names: Sequence[str],
    bg_name: str,
    tgt_name: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    names = list(source_names)
    y_train = y[splits["stack_train"]]
    y_val = y[splits["stack_val"]]
    y_test = y[splits["final_test"]]

    logits_train = [logits_by_source[n][splits["stack_train"]] for n in names]
    logits_val = [logits_by_source[n][splits["stack_val"]] for n in names]
    logits_test = [logits_by_source[n][splits["final_test"]] for n in names]
    probs_train = [_softmax(z) for z in logits_train]
    probs_val = [_softmax(z) for z in logits_val]
    probs_test = [_softmax(z) for z in logits_test]

    methods: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]] = {}
    uni = np.full((len(names),), 1.0 / len(names), dtype=np.float32)
    methods["uniform_prob_avg"] = (
        _fuse_probs(uni, probs_train),
        _fuse_probs(uni, probs_val),
        _fuse_probs(uni, probs_test),
        {"weights": {n: float(w) for n, w in zip(names, uni)}},
    )

    wp, wp_acc = _search_weights(y_train, probs_train, "prob", int(args.weight_random_samples), int(args.seed) + 11)
    methods["weighted_prob_avg"] = (
        _fuse_probs(wp, probs_train),
        _fuse_probs(wp, probs_val),
        _fuse_probs(wp, probs_test),
        {"train_search_acc": wp_acc, "weights": {n: float(w) for n, w in zip(names, wp)}},
    )

    wl, wl_acc = _search_weights(y_train, logits_train, "logit", int(args.weight_random_samples), int(args.seed) + 12)
    methods["weighted_logit_avg"] = (
        _softmax(_fuse_logits(wl, logits_train)),
        _softmax(_fuse_logits(wl, logits_val)),
        _softmax(_fuse_logits(wl, logits_test)),
        {"train_search_acc": wl_acc, "weights": {n: float(w) for n, w in zip(names, wl)}},
    )

    x_train = _stack_features(logits_train, probs_train, str(args.stack_features))
    x_val = _stack_features(logits_val, probs_val, str(args.stack_features))
    x_test = _stack_features(logits_test, probs_test, str(args.stack_features))
    stack_model, info_train = _fit_logreg_model(x_train, y_train, args)
    p_train = stack_model.predict_proba(x_train).astype(np.float32)
    p_val = stack_model.predict_proba(x_val).astype(np.float32)
    p_test = stack_model.predict_proba(x_test).astype(np.float32)
    methods["stacked_logreg"] = (p_train, p_val, p_test, info_train)

    method_metrics: Dict[str, object] = {}
    best_method = None
    best_val_acc = -1.0
    for method, (p_tr, p_va, p_te, info) in methods.items():
        mt = _metrics(y_train, p_tr, class_names, bg_name, tgt_name)
        mv = _metrics(y_val, p_va, class_names, bg_name, tgt_name)
        mf = _metrics(y_test, p_te, class_names, bg_name, tgt_name)
        method_metrics[method] = {"stack_train": mt, "stack_val": mv, "final_test": mf, "info": info}
        if float(mv["acc"]) > best_val_acc:
            best_val_acc = float(mv["acc"])
            best_method = method

    controls: List[Dict[str, object]] = []
    if bool(args.run_controls):
        rng = np.random.default_rng(int(args.seed) + 200)
        y_perm = y_train.copy()
        rng.shuffle(y_perm)
        control_model, _ = _fit_logreg_model(x_train, y_perm, args)
        p_control = control_model.predict_proba(x_test).astype(np.float32)
        controls.append({
            "group": group_name,
            "control": "permuted_stack_train_labels",
            "final_test": _metrics(y_test, p_control, class_names, bg_name, tgt_name),
        })
        x_shuf = x_train.copy()
        for c in range(x_shuf.shape[1]):
            rng.shuffle(x_shuf[:, c])
        control_model, _ = _fit_logreg_model(x_shuf, y_train, args)
        p_control = control_model.predict_proba(x_test).astype(np.float32)
        controls.append({
            "group": group_name,
            "control": "row_shuffled_stack_train_features",
            "final_test": _metrics(y_test, p_control, class_names, bg_name, tgt_name),
        })

    return {
        "group": group_name,
        "source_names": names,
        "method_metrics": method_metrics,
        "best_method_by_stack_val_acc": best_method,
    }, controls


def _evaluate_singletons(
    logits_by_source: Dict[str, np.ndarray],
    y: np.ndarray,
    splits: Dict[str, slice],
    class_names: Sequence[str],
    bg_name: str,
    tgt_name: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    y_train = y[splits["stack_train"]]
    y_val = y[splits["stack_val"]]
    y_test = y[splits["final_test"]]
    for name, logits in logits_by_source.items():
        z_tr = logits[splits["stack_train"]]
        z_va = logits[splits["stack_val"]]
        z_te = logits[splits["final_test"]]
        p_tr = _softmax(z_tr)
        p_va = _softmax(z_va)
        p_te = _softmax(z_te)
        raw = {
            "stack_train": _metrics(y_train, p_tr, class_names, bg_name, tgt_name),
            "stack_val": _metrics(y_val, p_va, class_names, bg_name, tgt_name),
            "final_test": _metrics(y_test, p_te, class_names, bg_name, tgt_name),
        }
        x_tr = _stack_features([z_tr], [p_tr], str(args.stack_features))
        x_va = _stack_features([z_va], [p_va], str(args.stack_features))
        x_te = _stack_features([z_te], [p_te], str(args.stack_features))
        stack_model, info = _fit_logreg_model(x_tr, y_train, args)
        p_stack_tr = stack_model.predict_proba(x_tr).astype(np.float32)
        p_stack_va = stack_model.predict_proba(x_va).astype(np.float32)
        p_stack_te = stack_model.predict_proba(x_te).astype(np.float32)
        stacked = {
            "stack_train": _metrics(y_train, p_stack_tr, class_names, bg_name, tgt_name),
            "stack_val": _metrics(y_val, p_stack_va, class_names, bg_name, tgt_name),
            "final_test": _metrics(y_test, p_stack_te, class_names, bg_name, tgt_name),
            "info": info,
        }
        out[name] = {"raw": raw, "singleton_stacked_logreg": stacked}
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="name:kind:run_dir")
    ap.add_argument("--stack_group", action="append", default=[], help="group:name1,name2,...")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--source_split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--stack_train_jets", type=int, required=True)
    ap.add_argument("--stack_val_jets", type=int, required=True)
    ap.add_argument("--final_test_jets", type=int, required=True)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--fresh_data_seed_offset", type=int, default=7001)
    ap.add_argument("--hlt_seed_offset", type=int, default=8001)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--stack_features", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    ap.add_argument("--stack_cv", type=int, default=5)
    ap.add_argument("--stack_Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--stack_max_iter", type=int, default=2000)
    ap.add_argument("--stack_n_jobs", type=int, default=1)
    ap.add_argument("--weight_random_samples", type=int, default=2500)
    ap.add_argument("--run_controls", action="store_true")
    ap.add_argument("--hash_audit_rows", type=int, default=20000)
    ap.add_argument("--save_scores", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    specs = [_parse_source(s) for s in args.source]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("Duplicate source names are not allowed.")
    for s in specs:
        if not (s.run_dir / "args.json").exists():
            raise FileNotFoundError(f"Missing args.json for source `{s.name}`: {s.run_dir}")
    run_args = {s.name: _ns(s.run_dir / "args.json") for s in specs}
    ref = run_args[specs[0].name]
    for s in specs[1:]:
        _check_compat(ref, run_args[s.name], s.name)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = int(args.stack_train_jets) + int(args.stack_val_jets) + int(args.final_test_jets)
    print("=" * 72)
    print("Fresh JetClass Fusion Audit")
    print(f"Out: {out_dir}")
    print(f"Data: {args.data_dir}")
    print(f"Source split: {args.source_split}")
    print(f"Splits: stack_train={args.stack_train_jets} stack_val={args.stack_val_jets} final_test={args.final_test_jets}")
    print(f"Device: {device}")
    print("=" * 72)

    class_names, off_tok, off_mask, y = _load_fresh_source_events(
        ref,
        args.data_dir,
        args.source_split,
        total,
        int(ref.seed) + int(args.fresh_data_seed_offset),
    )
    splits = _split_indices(args.stack_train_jets, args.stack_val_jets, args.final_test_jets)
    hash_report = _hash_audit(off_tok, off_mask, splits, int(args.hash_audit_rows), int(args.seed) + 444)
    print(f"Hash audit: {hash_report}")

    print("Building fixed HLT view for fresh source events...")
    hlt_params = HLTParams(
        hlt_pt_threshold=float(ref.hlt_pt_threshold),
        merge_prob_scale=float(ref.merge_prob_scale),
        reassign_scale=float(ref.reassign_scale),
        smear_scale=float(ref.smear_scale),
        eff_plateau_barrel=float(ref.eff_plateau_barrel),
        eff_plateau_endcap=float(ref.eff_plateau_endcap),
        eff_turnon_pt=float(ref.eff_turnon_pt),
        eff_width_pt=float(ref.eff_width_pt),
    )
    hlt_tok, hlt_mask, _ = build_hlt_view(
        off_tok,
        off_mask,
        params=hlt_params,
        seed=int(ref.seed) + int(args.hlt_seed_offset),
    )

    print("Computing offline and HLT features...")
    feat_off, standardization_mode = _feature_arrays(ref, off_tok, off_mask, args.data_dir)
    feat_hlt = compute_features(hlt_tok, hlt_mask, str(ref.feature_mode), str(ref.feature_preprocessing))
    if standardization_mode == "learned_train_split":
        # _feature_arrays already standardizes offline. Recompute train stats for HLT.
        # This path is not expected for the fixed-HLT filename runs, which use canonical preprocessing.
        raise RuntimeError("Non-canonical preprocessing is not supported by this fresh audit path yet.")

    ds = FreshEvalDataset(feat_hlt, hlt_mask, hlt_tok[:, :, :4].astype(np.float32), feat_off, off_mask, y)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    input_dim = int(feat_hlt.shape[-1])
    n_classes = int(len(class_names))
    loaded: List[LoadedSource] = []
    for spec in specs:
        print(f"Loading source {spec.name} ({spec.kind}) from {spec.run_dir}")
        loaded.append(_load_source(spec, ref, input_dim, n_classes, device, float(args.corrected_weight_floor)))

    logits_by_source: Dict[str, np.ndarray] = {}
    y_ref = None
    for src in loaded:
        print(f"Collecting logits: {src.name}")
        logits, yy = _collect_logits(src, loader)
        if y_ref is None:
            y_ref = yy
        elif not np.array_equal(y_ref, yy):
            raise RuntimeError(f"Label mismatch while collecting logits for {src.name}")
        logits_by_source[src.name] = logits.astype(np.float32)

    # Fit temperature per source on stack_train only.
    temperatures: Dict[str, Dict[str, float]] = {}
    st = splits["stack_train"]
    y_train = y_ref[st]
    for name, logits in list(logits_by_source.items()):
        t, before, after = _fit_temperature(logits[st], y_train)
        temperatures[name] = {"temperature": t, "train_nll_before": before, "train_nll_after": after}
        logits_by_source[name] = (logits / float(t)).astype(np.float32)

    bg_name = str(ref.background_class)
    tgt_name = str(ref.target_class)
    singleton = _evaluate_singletons(logits_by_source, y_ref, splits, class_names, bg_name, tgt_name, args)

    groups = []
    if args.stack_group:
        groups = [_parse_stack_group(s) for s in args.stack_group]
    else:
        names = [s.name for s in specs if s.kind != "offline_teacher"]
        groups = [("all_non_teacher", names)]

    group_reports = []
    control_reports = []
    for group_name, names in groups:
        missing = [n for n in names if n not in logits_by_source]
        if missing:
            raise ValueError(f"Stack group `{group_name}` references missing sources: {missing}")
        print(f"Evaluating stack group {group_name}: {', '.join(names)}")
        report, controls = _evaluate_group(group_name, names, logits_by_source, y_ref, splits, class_names, bg_name, tgt_name, args)
        group_reports.append(report)
        control_reports.extend(controls)

    source_rows = []
    for name, vals in singleton.items():
        source_rows.append({
            "source": name,
            "raw_stack_val_acc": vals["raw"]["stack_val"]["acc"],
            "raw_final_test_acc": vals["raw"]["final_test"]["acc"],
            "singleton_stack_val_acc": vals["singleton_stacked_logreg"]["stack_val"]["acc"],
            "singleton_final_test_acc": vals["singleton_stacked_logreg"]["final_test"]["acc"],
        })
    _write_csv(out_dir / "singleton_summary.csv", source_rows)

    group_rows = []
    for gr in group_reports:
        for method, vals in gr["method_metrics"].items():
            group_rows.append({
                "group": gr["group"],
                "method": method,
                "stack_train_acc": vals["stack_train"]["acc"],
                "stack_val_acc": vals["stack_val"]["acc"],
                "final_test_acc": vals["final_test"]["acc"],
                "stack_val_auc": vals["stack_val"]["auc_macro_ovr"],
                "final_test_auc": vals["final_test"]["auc_macro_ovr"],
            })
    _write_csv(out_dir / "group_method_summary.csv", group_rows)

    report = {
        "protocol": {
            "source_split": str(args.source_split),
            "stack_train_jets": int(args.stack_train_jets),
            "stack_val_jets": int(args.stack_val_jets),
            "final_test_jets": int(args.final_test_jets),
            "stacker_fit_split": "stack_train only",
            "method_selection_split": "stack_val only",
            "final_test_policy": "final_test labels are only used for final reporting",
            "stack_features": str(args.stack_features),
        },
        "class_names": list(class_names),
        "standardization_mode": standardization_mode,
        "reference_run": str(specs[0].run_dir),
        "reference_args_subset": {
            "class_assignment": str(ref.class_assignment),
            "feature_preprocessing": str(ref.feature_preprocessing),
            "feature_mode": str(ref.feature_mode),
            "seed": int(ref.seed),
            "hlt_pt_threshold": float(ref.hlt_pt_threshold),
            "merge_prob_scale": float(ref.merge_prob_scale),
            "reassign_scale": float(ref.reassign_scale),
            "smear_scale": float(ref.smear_scale),
            "eff_plateau_barrel": float(ref.eff_plateau_barrel),
            "eff_plateau_endcap": float(ref.eff_plateau_endcap),
            "eff_turnon_pt": float(ref.eff_turnon_pt),
            "eff_width_pt": float(ref.eff_width_pt),
        },
        "sources": [{"name": s.name, "kind": s.kind, "run_dir": str(s.run_dir), "loaded": src.sources} for s, src in zip(specs, loaded)],
        "hash_audit": hash_report,
        "temperatures": temperatures,
        "singleton_metrics": singleton,
        "group_reports": group_reports,
        "controls": control_reports,
    }
    (out_dir / "fresh_fusion_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if bool(args.save_scores):
        np.savez_compressed(
            out_dir / "fresh_fusion_logits.npz",
            y=y_ref.astype(np.int64),
            **{f"logits_{name}": z.astype(np.float32) for name, z in logits_by_source.items()},
        )

    print("-" * 72)
    print("Singleton summary")
    for row in source_rows:
        print(
            f"{row['source']:18s} raw_test={row['raw_final_test_acc']:.6f} "
            f"singleton_stack_test={row['singleton_final_test_acc']:.6f}"
        )
    print("Group summary")
    for row in group_rows:
        print(
            f"{row['group']:16s} {row['method']:20s} "
            f"val={row['stack_val_acc']:.6f} test={row['final_test_acc']:.6f}"
        )
    for gr in group_reports:
        print(f"Best for {gr['group']} by stack_val: {gr['best_method_by_stack_val_acc']}")
    print(f"Saved report: {out_dir / 'fresh_fusion_audit_report.json'}")


if __name__ == "__main__":
    main()
