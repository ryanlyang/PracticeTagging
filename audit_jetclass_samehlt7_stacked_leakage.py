#!/usr/bin/env python3
"""Audit the JetClass same-HLT stacked-logreg result for leakage-like failures.

The main path consumes an existing fusion report/NPZ so the expensive 1M-jet
model inference is not repeated. Optional input controls reload the same models
on a small sample and verify the stacker reacts to HLT inputs, while there is no
offline tensor path in the eval dataset after HLT corruption is built.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import analyze_jetclass_four_model_stacked_fusion as fusion


SUSPICIOUS_SOURCE_TOKENS = (
    "teacher",
    "fused",
    "target_scores",
    "oracle",
    "offline_logits",
    "teacher_logits",
    "fused_targets",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--report_json",
        type=Path,
        default=Path(
            "checkpoints/jetclass_joint_dualview/fusion_reports/"
            "samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/report.json"
        ),
    )
    p.add_argument(
        "--scores_npz",
        type=Path,
        default=Path(
            "checkpoints/jetclass_joint_dualview/fusion_reports/"
            "samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/fusion_scores.npz"
        ),
    )
    p.add_argument(
        "--hlt5_report_json",
        type=Path,
        default=Path(
            "checkpoints/jetclass_hlt_seed_ensemble/fusion_reports/"
            "hlt5_1m250k1m_fixedhlt_stacked_acc/report.json"
        ),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "checkpoints/jetclass_joint_dualview/fusion_reports/"
            "samehlt7_plus_hlt_1m250k1m_m2hybrid_stacked_acc/leakage_audit"
        ),
    )
    p.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    p.add_argument("--feature_mode", default="logits_probs", choices=["logits", "probs", "logits_probs"])
    p.add_argument("--stack_cv", type=int, default=5)
    p.add_argument("--stack_max_iter", type=int, default=2000)
    p.add_argument("--stack_n_jobs", type=int, default=1)
    p.add_argument("--stack_Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--holdout_repeats", type=int, default=3)
    p.add_argument("--holdout_train_frac", type=float, default=0.5)
    p.add_argument("--random_seed", type=int, default=52)
    p.add_argument("--hash_jets_per_split", type=int, default=20000)
    p.add_argument("--run_input_controls", action="store_true", default=True)
    p.add_argument("--skip_input_controls", dest="run_input_controls", action="store_false")
    p.add_argument("--input_control_jets", type=int, default=20000)
    p.add_argument("--input_control_batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _model_names(report: Dict) -> List[str]:
    return [str(m["name"]) for m in report["models"]]


def _safe(name: str) -> str:
    return name.replace(" ", "_")


def _build_stack_features(d: np.lib.npyio.NpzFile, names: Sequence[str], split: str, mode: str) -> np.ndarray:
    logits = [d[f"logits_{split}_{_safe(n)}"].astype(np.float32) for n in names]
    probs = [d[f"probs_{split}_{_safe(n)}"].astype(np.float32) for n in names]
    if mode == "logits":
        return np.concatenate(logits, axis=1).astype(np.float32)
    if mode == "probs":
        return np.concatenate(probs, axis=1).astype(np.float32)
    return np.concatenate([np.concatenate(logits, axis=1), np.concatenate(probs, axis=1)], axis=1).astype(np.float32)


def _build_stack_features_from_arrays(
    logits_list: Sequence[np.ndarray],
    probs_list: Sequence[np.ndarray],
    mode: str,
) -> np.ndarray:
    if mode == "logits":
        return np.concatenate(logits_list, axis=1).astype(np.float32)
    if mode == "probs":
        return np.concatenate(probs_list, axis=1).astype(np.float32)
    return np.concatenate(
        [np.concatenate(logits_list, axis=1), np.concatenate(probs_list, axis=1)],
        axis=1,
    ).astype(np.float32)


def _fit_stack(x: np.ndarray, y: np.ndarray, args: argparse.Namespace):
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
    clf.fit(x, y)
    return clf


def _metrics(y: np.ndarray, p: np.ndarray, class_names: Sequence[str], bg: str, tgt: str) -> Dict[str, float]:
    return fusion._to_float_dict(fusion.eval_metrics(y, p, class_names, bg, tgt))


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def _source_audit(report: Dict) -> Dict:
    suspicious = []
    for m in report["models"]:
        blob = json.dumps({"run_dir": m.get("run_dir"), "sources": m.get("sources", {})}, sort_keys=True).lower()
        hits = [tok for tok in SUSPICIOUS_SOURCE_TOKENS if tok in blob]
        if hits:
            suspicious.append({"model": m["name"], "hits": hits, "sources": m.get("sources", {})})
    return {
        "pass": len(suspicious) == 0,
        "suspicious": suspicious,
        "checked_tokens": list(SUSPICIOUS_SOURCE_TOKENS),
    }


def _args_for_report_models(report: Dict) -> Dict[str, SimpleNamespace]:
    out = {}
    for m in report["models"]:
        p = Path(m["run_dir"]) / "args.json"
        out[str(m["name"])] = SimpleNamespace(**json.loads(p.read_text()))
    return out


def _hlt_compat_audit(args_map: Dict[str, SimpleNamespace]) -> Dict:
    names = list(args_map)
    ref = args_map[names[0]]
    keys = (
        "seed",
        "feature_mode",
        "feature_preprocessing",
        "class_assignment",
        "n_train_jets",
        "n_val_jets",
        "n_test_jets",
        "max_constits",
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
    for name, ns in args_map.items():
        for k in keys:
            if str(getattr(ref, k, None)) != str(getattr(ns, k, None)):
                mismatches.append(
                    {"model": name, "key": k, "ref": getattr(ref, k, None), "value": getattr(ns, k, None)}
                )
    return {"pass": len(mismatches) == 0, "reference_model": names[0], "mismatches": mismatches}


def _split_file_audit(ref_args: SimpleNamespace, data_dir: Path) -> Dict:
    files_by_class = fusion.collect_files_by_class(data_dir.resolve())
    if str(ref_args.class_assignment) == "canonical_labels":
        class_names = list(fusion.CANONICAL_CLASS_ORDER)
        source_files = files_by_class
    else:
        class_names = sorted(files_by_class.keys())
        source_files = {c: files_by_class[c] for c in class_names}
    tr, va, te = fusion.split_files_by_class(
        source_files,
        n_train=int(ref_args.train_files_per_class),
        n_val=int(ref_args.val_files_per_class),
        n_test=int(ref_args.test_files_per_class),
        shuffle=bool(ref_args.shuffle_files),
        seed=int(ref_args.seed),
    )

    def as_path(x) -> Path:
        # collect_files_by_class stores (file_index, Path), while split_files_by_class
        # returns bare Paths. Normalize both forms before overlap checks.
        if isinstance(x, tuple):
            x = x[-1]
        return Path(x).resolve()

    def flat(d):
        return {str(as_path(p)) for xs in d.values() for p in xs}

    ftr, fva, fte = flat(tr), flat(va), flat(te)
    all_used = ftr | fva | fte
    all_files = flat(source_files)
    unused_by_class = {
        c: [str(as_path(p)) for p in source_files[c] if str(as_path(p)) not in all_used]
        for c in class_names
    }
    return {
        "pass": not (ftr & fva or ftr & fte or fva & fte),
        "n_train_files": len(ftr),
        "n_val_files": len(fva),
        "n_test_files": len(fte),
        "train_val_overlap": sorted(ftr & fva),
        "train_test_overlap": sorted(ftr & fte),
        "val_test_overlap": sorted(fva & fte),
        "unused_files_per_class": {k: len(v) for k, v in unused_by_class.items()},
        "independent_unused_file_sample_available": any(len(v) > 0 for v in unused_by_class.values()),
    }


def _hash_rows(tok: np.ndarray, mask: np.ndarray) -> set[str]:
    hashes = set()
    tok_q = np.round(tok.astype(np.float32), 5)
    for i in range(tok_q.shape[0]):
        h = hashlib.blake2b(digest_size=16)
        h.update(mask[i].astype(np.uint8).tobytes())
        h.update(tok_q[i].tobytes())
        hashes.add(h.hexdigest())
    return hashes


def _sample_hash_audit(ref_args: SimpleNamespace, data_dir: Path, n_each: int) -> Dict:
    if n_each <= 0:
        return {"skipped": True, "reason": "hash_jets_per_split <= 0"}
    files_by_class = fusion.collect_files_by_class(data_dir.resolve())
    if str(ref_args.class_assignment) == "canonical_labels":
        class_names = list(fusion.CANONICAL_CLASS_ORDER)
        source_files = files_by_class
    else:
        class_names = sorted(files_by_class.keys())
        source_files = {c: files_by_class[c] for c in class_names}
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    tr, va, te = fusion.split_files_by_class(
        source_files,
        n_train=int(ref_args.train_files_per_class),
        n_val=int(ref_args.val_files_per_class),
        n_test=int(ref_args.test_files_per_class),
        shuffle=bool(ref_args.shuffle_files),
        seed=int(ref_args.seed),
    )
    tr_tok, tr_mask, _ = fusion.load_split(
        tr,
        n_total=int(n_each),
        max_constits=int(ref_args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(ref_args.seed) + 101,
        class_assignment=str(ref_args.class_assignment),
    )
    va_tok, va_mask, _ = fusion.load_split(
        va,
        n_total=int(n_each),
        max_constits=int(ref_args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(ref_args.seed) + 202,
        class_assignment=str(ref_args.class_assignment),
    )
    te_tok, te_mask, _ = fusion.load_split(
        te,
        n_total=int(n_each),
        max_constits=int(ref_args.max_constits),
        class_to_idx=class_to_idx,
        seed=int(ref_args.seed) + 303,
        class_assignment=str(ref_args.class_assignment),
    )
    htr, hva, hte = _hash_rows(tr_tok, tr_mask), _hash_rows(va_tok, va_mask), _hash_rows(te_tok, te_mask)
    return {
        "skipped": False,
        "n_each": int(n_each),
        "train_val_hash_overlap": int(len(htr & hva)),
        "train_test_hash_overlap": int(len(htr & hte)),
        "val_test_hash_overlap": int(len(hva & hte)),
        "pass": len(htr & hva) == 0 and len(htr & hte) == 0 and len(hva & hte) == 0,
        "note": "sampled constituent-array hash audit; file-level audit is stricter for split assignment",
    }


def _stacking_controls(
    report: Dict,
    scores_npz: Path,
    args: argparse.Namespace,
) -> Tuple[Dict, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(scores_npz)
    names = _model_names(report)
    class_names = report["class_names"]
    bg = str(report["background_class"])
    tgt = str(report["target_class"])
    y_val = d["y_val"].astype(np.int64)
    y_test = d["y_test"].astype(np.int64)
    x_val = _build_stack_features(d, names, "val", args.feature_mode)
    x_test = _build_stack_features(d, names, "test", args.feature_mode)

    stack = _fit_stack(x_val, y_val, args)
    p_val = stack.predict_proba(x_val).astype(np.float32)
    p_test = stack.predict_proba(x_test).astype(np.float32)

    rng = np.random.default_rng(int(args.random_seed))
    y_perm = y_val.copy()
    rng.shuffle(y_perm)
    perm_stack = _fit_stack(x_val, y_perm, args)
    p_perm_test = perm_stack.predict_proba(x_test).astype(np.float32)

    idx = rng.permutation(len(y_val))
    n_train = int(round(float(args.holdout_train_frac) * len(idx)))
    n_train = max(1, min(n_train, len(idx) - 1))
    holdouts = []
    for rep in range(int(args.holdout_repeats)):
        idx = np.random.default_rng(int(args.random_seed) + 1000 + rep).permutation(len(y_val))
        tr_idx = idx[:n_train]
        ho_idx = idx[n_train:]
        hold_stack = _fit_stack(x_val[tr_idx], y_val[tr_idx], args)
        p_ho = hold_stack.predict_proba(x_val[ho_idx]).astype(np.float32)
        p_te = hold_stack.predict_proba(x_test).astype(np.float32)
        holdouts.append(
            {
                "repeat": rep,
                "meta_train_size": int(len(tr_idx)),
                "meta_holdout_size": int(len(ho_idx)),
                "holdout": _metrics(y_val[ho_idx], p_ho, class_names, bg, tgt),
                "test": _metrics(y_test, p_te, class_names, bg, tgt),
            }
        )

    # Fit on true val, then damage each model block at test time only.
    n_classes = len(class_names)
    per_model_blocks = []
    for i, name in enumerate(names):
        damaged = x_test.copy()
        if args.feature_mode in {"logits", "probs"}:
            starts = [i * n_classes]
            width = n_classes
        else:
            starts = [i * n_classes, len(names) * n_classes + i * n_classes]
            width = n_classes
        perm = np.random.default_rng(int(args.random_seed) + 2000 + i).permutation(x_test.shape[0])
        for st in starts:
            damaged[:, st : st + width] = damaged[perm, st : st + width]
        p_damaged = stack.predict_proba(damaged).astype(np.float32)
        per_model_blocks.append(
            {
                "model": name,
                "test_acc_after_row_shuffle_this_model": _acc(y_test, p_damaged),
                "acc_drop": _acc(y_test, p_test) - _acc(y_test, p_damaged),
            }
        )

    all_damaged = x_test.copy()
    perm = np.random.default_rng(int(args.random_seed) + 3000).permutation(x_test.shape[0])
    all_damaged[:, :] = all_damaged[perm, :]
    p_all_damaged = stack.predict_proba(all_damaged).astype(np.float32)

    result = {
        "full_val_fit": {
            "val": _metrics(y_val, p_val, class_names, bg, tgt),
            "test": _metrics(y_test, p_test, class_names, bg, tgt),
        },
        "permuted_val_labels": {
            "test": _metrics(y_test, p_perm_test, class_names, bg, tgt),
            "expected": "near random for 10 classes if no label leakage",
        },
        "meta_holdout_repeats": holdouts,
        "row_shuffle_test_only_each_model_block": per_model_blocks,
        "row_shuffle_test_only_all_features": {
            "test": _metrics(y_test, p_all_damaged, class_names, bg, tgt),
            "expected": "large collapse if per-jet model outputs matter",
        },
    }
    return result, stack, x_val, y_val, x_test, y_test


def _hlt5_control(report_path: Path) -> Dict:
    if not report_path.exists():
        return {"available": False, "path": str(report_path)}
    r = _load_json(report_path)
    stacked = r.get("method_metrics", {}).get("stacked_logreg", {}).get("test", {})
    best = None
    for name, m in r.get("individual_metrics", {}).items():
        t = m.get("test", {})
        if best is None or float(t.get("acc", -1.0)) > float(best[1].get("acc", -1.0)):
            best = (name, t)
    return {
        "available": True,
        "path": str(report_path),
        "stacked_logreg_test": stacked,
        "best_individual_test": {"name": best[0], "metrics": best[1]} if best else None,
    }


def _input_controls(
    report: Dict,
    args_map: Dict[str, SimpleNamespace],
    stack,
    args: argparse.Namespace,
) -> Dict:
    if not args.run_input_controls:
        return {"skipped": True, "reason": "--skip_input_controls"}
    if fusion._IMPORT_ERROR is not None:
        return {"skipped": True, "reason": f"fusion import error: {fusion._IMPORT_ERROR}"}

    ref_args = copy.deepcopy(args_map[_model_names(report)[0]])
    ref_args.n_train_jets = min(int(getattr(ref_args, "n_train_jets", args.input_control_jets)), int(args.input_control_jets))
    ref_args.n_val_jets = int(args.input_control_jets)
    ref_args.n_test_jets = int(args.input_control_jets)

    (
        class_names,
        va_feat_hlt,
        va_hlt_mask,
        va_hlt_const4,
        va_y,
        te_feat_hlt,
        te_hlt_mask,
        te_hlt_const4,
        te_y,
        _standardization_mode,
    ) = fusion._build_eval_data(ref_args, args.data_dir)

    # This is the core offline-leakage structural check: the eval dataset has no
    # offline/offline-target tensor once the HLT view has been built.
    sample_ds = fusion.JointEvalDataset(te_feat_hlt[:1], te_hlt_mask[:1], te_hlt_const4[:1], te_y[:1])
    dataset_keys = sorted(list(sample_ds[0].keys()))
    contains_offline_key = any("off" in k.lower() or "teacher" in k.lower() for k in dataset_keys)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    specs = [fusion._parse_model_spec(f"{m['name']}:{m['kind']}:{m['run_dir']}") for m in report["models"]]
    input_dim = int(va_feat_hlt.shape[-1])
    n_classes = len(class_names)
    loaded = [
        fusion._load_model_for_spec(
            s,
            args_map[s.name],
            input_dim=input_dim,
            n_classes=n_classes,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
        )
        for s in specs
    ]

    dl_te = DataLoader(
        fusion.JointEvalDataset(te_feat_hlt, te_hlt_mask, te_hlt_const4, te_y),
        batch_size=int(args.input_control_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_zero = DataLoader(
        fusion.JointEvalDataset(
            np.zeros_like(te_feat_hlt),
            te_hlt_mask.copy(),
            np.zeros_like(te_hlt_const4),
            te_y,
        ),
        batch_size=int(args.input_control_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    temps = report.get("temperature_per_model", {})
    logits_orig, logits_zero = [], []
    for lm in loaded:
        lo, y_ref = fusion._collect_logits(lm, dl_te)
        lz, yz = fusion._collect_logits(lm, dl_zero)
        if not np.array_equal(y_ref, yz):
            raise RuntimeError("Input-control labels changed unexpectedly.")
        t = float(temps.get(lm.name, 1.0))
        logits_orig.append((lo / t).astype(np.float32))
        logits_zero.append((lz / t).astype(np.float32))
    probs_orig = [fusion._softmax_np(z) for z in logits_orig]
    probs_zero = [fusion._softmax_np(z) for z in logits_zero]
    x_orig = _build_stack_features_from_arrays(logits_orig, probs_orig, args.feature_mode)
    x_zero = _build_stack_features_from_arrays(logits_zero, probs_zero, args.feature_mode)
    p_orig = stack.predict_proba(x_orig).astype(np.float32)
    p_zero = stack.predict_proba(x_zero).astype(np.float32)

    return {
        "skipped": False,
        "n_test_jets": int(len(te_y)),
        "eval_dataset_keys": dataset_keys,
        "contains_offline_or_teacher_key": bool(contains_offline_key),
        "original_hlt_sample": _metrics(te_y, p_orig, class_names, str(report["background_class"]), str(report["target_class"])),
        "zero_hlt_features_and_const4_sample": _metrics(
            te_y, p_zero, class_names, str(report["background_class"]), str(report["target_class"])
        ),
        "mean_abs_stack_prob_delta_zero_vs_original": float(np.mean(np.abs(p_orig - p_zero))),
        "expected": "zero-HLT should degrade/change; dataset keys should contain no offline/teacher tensors",
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = _load_json(args.report_json)
    args_map = _args_for_report_models(report)
    names = _model_names(report)

    print("============================================================")
    print("JetClass Same-HLT 7+HLT Stacked Leakage Audit")
    print("============================================================")
    print(f"Report: {args.report_json}")
    print(f"Scores: {args.scores_npz}")
    print(f"Models ({len(names)}): {', '.join(names)}")

    source_audit = _source_audit(report)
    hlt_compat = _hlt_compat_audit(args_map)
    split_files = _split_file_audit(args_map[names[0]], args.data_dir)
    hash_audit = _sample_hash_audit(args_map[names[0]], args.data_dir, int(args.hash_jets_per_split))
    stacking_controls, stack, _xv, _yv, _xt, _yt = _stacking_controls(report, args.scores_npz, args)
    input_controls = _input_controls(report, args_map, stack, args)
    hlt5 = _hlt5_control(args.hlt5_report_json)

    result = {
        "report_json": str(args.report_json.resolve()),
        "scores_npz": str(args.scores_npz.resolve()),
        "models": report["models"],
        "source_audit": source_audit,
        "hlt_compat_audit": hlt_compat,
        "split_file_audit": split_files,
        "sample_hash_audit": hash_audit,
        "stacking_controls": stacking_controls,
        "input_controls": input_controls,
        "hlt5_seed_ensemble_control": hlt5,
        "interpretation": {
            "strong_pass_conditions": [
                "source_audit.pass is true",
                "hlt_compat_audit.pass is true for same-HLT claim",
                "split_file_audit.pass is true",
                "sample_hash_audit.pass is true or no overlaps in sampled hashes",
                "permuted_val_labels test accuracy collapses near chance",
                "meta-holdout test accuracy remains close to full-val stacked test accuracy",
                "row-shuffled feature controls degrade materially",
                "input_controls show no offline/teacher keys and zero-HLT changes/degrades predictions",
                "HLT5 seed ensemble stacked result is meaningfully below reconstructor-diverse stack",
            ],
            "caveat": "This does not solve synthetic-HLT-vs-real-HLT domain shift; it audits leakage in this offline simulation pipeline.",
        },
    }

    out_json = args.out_dir / "leakage_audit_report.json"
    out_json.write_text(json.dumps(result, indent=2))

    full = stacking_controls["full_val_fit"]["test"]
    perm = stacking_controls["permuted_val_labels"]["test"]
    all_shuf = stacking_controls["row_shuffle_test_only_all_features"]["test"]
    print("------------------------------------------------------------")
    print(f"Source audit pass: {source_audit['pass']}")
    print(f"Same-HLT compat pass: {hlt_compat['pass']}")
    print(f"Split file audit pass: {split_files['pass']}")
    print(f"Sample hash audit pass: {hash_audit.get('pass', 'skipped')}")
    print(
        "Stack full-val-fit test: "
        f"acc={float(full.get('acc', float('nan'))):.6f} "
        f"auc={float(full.get('auc_macro_ovr', float('nan'))):.6f} "
        f"fpr50={float(full.get('signal_vs_bg_fpr50', float('nan'))):.6f}"
    )
    print(
        "Permuted-label control test: "
        f"acc={float(perm.get('acc', float('nan'))):.6f} "
        f"auc={float(perm.get('auc_macro_ovr', float('nan'))):.6f}"
    )
    print(
        "All-feature row-shuffle control test: "
        f"acc={float(all_shuf.get('acc', float('nan'))):.6f} "
        f"auc={float(all_shuf.get('auc_macro_ovr', float('nan'))):.6f}"
    )
    if hlt5.get("available"):
        h5 = hlt5["stacked_logreg_test"]
        print(
            "HLT5 seed-ensemble stacked test: "
            f"acc={float(h5.get('acc', float('nan'))):.6f} "
            f"auc={float(h5.get('auc_macro_ovr', float('nan'))):.6f}"
        )
    else:
        print("HLT5 seed-ensemble report not found; comparison skipped.")
    if not input_controls.get("skipped"):
        orig = input_controls["original_hlt_sample"]
        zero = input_controls["zero_hlt_features_and_const4_sample"]
        print(
            "Input-control original/zero-HLT sample acc: "
            f"{float(orig.get('acc', float('nan'))):.6f} / {float(zero.get('acc', float('nan'))):.6f}"
        )
        print(f"Eval dataset keys: {input_controls['eval_dataset_keys']}")
    else:
        print(f"Input controls skipped: {input_controls.get('reason')}")
    print(f"Saved audit report: {out_json}")


if __name__ == "__main__":
    main()
