#!/usr/bin/env python3
"""
Four-model JetClass fusion with calibration + weighted averaging + stacked logistic fusion.

Typical use:
  python -u analyze_jetclass_four_model_stacked_fusion.py \
    --model "baseline:baseline_hlt:/path/to/runA" \
    --model "path:stage2:/path/to/runB" \
    --model "v1hlt:stage2:/path/to/runC" \
    --model "autologit:stage2:/path/to/runD" \
    --data_dir /path/to/jetclass_part0 \
    --out_dir checkpoints/jetclass_joint_dualview/fusion_reports/four_model_foo

Model spec format:
  name:kind:run_dir
with kind in:
  - baseline_hlt
  - stage2
  - joint
  - reco_only_stagea
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

CANONICAL_CLASS_ORDER = ()
HLTParams = None
JetClassTransformer = None
collect_files_by_class = None
compute_features = None
eval_metrics = None
get_mean_std = None
load_split = None
split_files_by_class = None
standardize = None

_EVAL_IMPORT_ERROR = None
try:
    from evaluate_jetclass_hlt_teacher_baseline import (
        CANONICAL_CLASS_ORDER as _CANONICAL_CLASS_ORDER,
        HLTParams as _HLTParams,
        JetClassTransformer as _JetClassTransformer,
        collect_files_by_class as _collect_files_by_class,
        compute_features as _compute_features,
        eval_metrics as _eval_metrics,
        get_mean_std as _get_mean_std,
        load_split as _load_split,
        split_files_by_class as _split_files_by_class,
        standardize as _standardize,
    )
    CANONICAL_CLASS_ORDER = _CANONICAL_CLASS_ORDER
    HLTParams = _HLTParams
    JetClassTransformer = _JetClassTransformer
    collect_files_by_class = _collect_files_by_class
    compute_features = _compute_features
    eval_metrics = _eval_metrics
    get_mean_std = _get_mean_std
    load_split = _load_split
    split_files_by_class = _split_files_by_class
    standardize = _standardize
except Exception as _exc:  # pragma: no cover
    _EVAL_IMPORT_ERROR = _exc

_IMPORT_ERROR = None
try:
    import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly_confgen_ops as confgen_ops
    import train_jetclass_joint_dualview_stage2_unmergeonly_v2_attr as v2
    from offline_reconstructor_no_gt_local30kv2 import CONFIG as BASE_RECO_CONFIG
except Exception as _exc:  # pragma: no cover
    confgen_ops = None
    v2 = None
    BASE_RECO_CONFIG = None
    _IMPORT_ERROR = _exc


class JointEvalDataset(Dataset):
    def __init__(
        self,
        feat_hlt: np.ndarray,
        mask_hlt: np.ndarray,
        const_hlt4: np.ndarray,
        labels: np.ndarray,
    ):
        self.feat_hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.mask_hlt = torch.tensor(mask_hlt, dtype=torch.bool)
        self.const_hlt4 = torch.tensor(const_hlt4, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "feat_hlt": self.feat_hlt[i],
            "mask_hlt": self.mask_hlt[i],
            "const_hlt4": self.const_hlt4[i],
            "label": self.labels[i],
        }


@dataclass
class ModelSpec:
    name: str
    kind: str
    run_dir: Path


@dataclass
class LoadedModel:
    name: str
    kind: str
    run_dir: Path
    predict_logits: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    sources: Dict[str, str]
    args: SimpleNamespace


def _ns_from_json(path: Path) -> SimpleNamespace:
    return SimpleNamespace(**json.loads(path.read_text()))


def _get_attr(ns: SimpleNamespace, key: str, default):
    return getattr(ns, key, default)


def _parse_model_spec(s: str) -> ModelSpec:
    parts = s.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --model spec `{s}`. Expected format: name:kind:run_dir"
        )
    name = parts[0].strip()
    kind = parts[1].strip()
    run_dir = Path(parts[2].strip()).resolve()
    if not name:
        raise ValueError(f"Empty model name in spec `{s}`")
    if kind not in {"baseline_hlt", "stage2", "joint", "reco_only_stagea"}:
        raise ValueError(
            f"Unsupported model kind `{kind}` in spec `{s}`. "
            "Supported: baseline_hlt, stage2, joint, reco_only_stagea."
        )
    return ModelSpec(name=name, kind=kind, run_dir=run_dir)


def _load_state_dict_from_run_dir(
    run_dir: Path,
    names: Sequence[str],
) -> Tuple[Dict[str, torch.Tensor], Path]:
    tried: List[str] = []
    for n in names:
        p = (run_dir / n).resolve()
        tried.append(str(p))
        if not p.exists():
            continue
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"], p
        if isinstance(obj, dict):
            return obj, p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}. Tried: {tried}")


def _check_run_compat(ref: SimpleNamespace, other: SimpleNamespace, label: str) -> None:
    strict_keys = (
        "feature_mode",
        "feature_preprocessing",
        "class_assignment",
        "target_class",
        "background_class",
        "n_val_jets",
        "n_test_jets",
        "max_constits",
    )
    for k in strict_keys:
        if str(getattr(ref, k)) != str(getattr(other, k)):
            raise ValueError(
                f"Run mismatch with {label} for `{k}`: ref={getattr(ref, k)} vs other={getattr(other, k)}"
            )

    # HLT corruption profile differences are allowed for fusion ablation families.
    # We evaluate all models on the same reference HLT view (from `ref`) for comparability.
    hlt_keys = (
        "hlt_pt_threshold",
        "merge_prob_scale",
        "reassign_scale",
        "smear_scale",
        "eff_plateau_barrel",
        "eff_plateau_endcap",
        "eff_turnon_pt",
        "eff_width_pt",
    )
    mismatches: List[str] = []
    for k in hlt_keys:
        rv = getattr(ref, k, None)
        ov = getattr(other, k, None)
        if str(rv) != str(ov):
            mismatches.append(f"{k}: ref={rv} vs other={ov}")
    if mismatches:
        print(
            f"[compat-warning] HLT profile mismatch for {label}; "
            "using reference-run HLT profile for fusion eval. "
            + "; ".join(mismatches)
        )


def _build_eval_data(
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
    str,
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

    hlt_params = HLTParams(
        hlt_pt_threshold=float(args_ref.hlt_pt_threshold),
        merge_prob_scale=float(args_ref.merge_prob_scale),
        reassign_scale=float(args_ref.reassign_scale),
        smear_scale=float(args_ref.smear_scale),
        eff_plateau_barrel=float(args_ref.eff_plateau_barrel),
        eff_plateau_endcap=float(args_ref.eff_plateau_endcap),
        eff_turnon_pt=float(args_ref.eff_turnon_pt),
        eff_width_pt=float(args_ref.eff_width_pt),
    )
    va_hlt_tok_raw, va_hlt_mask_raw, _, _ = v2.build_hlt_view(
        va_tok_raw,
        va_mask_raw,
        params=hlt_params,
        seed=int(args_ref.seed) + 1002,
        return_provenance=True,
    )
    te_hlt_tok_raw, te_hlt_mask_raw, _, _ = v2.build_hlt_view(
        te_tok_raw,
        te_mask_raw,
        params=hlt_params,
        seed=int(args_ref.seed) + 1003,
        return_provenance=True,
    )

    va_feat_hlt = compute_features(
        va_hlt_tok_raw,
        va_hlt_mask_raw,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    te_feat_hlt = compute_features(
        te_hlt_tok_raw,
        te_hlt_mask_raw,
        feature_mode=str(args_ref.feature_mode),
        feature_preprocessing=str(args_ref.feature_preprocessing),
    )
    standardization_mode = "canonical_manual_fixed"
    if str(args_ref.feature_preprocessing) != "canonical":
        tr_feat_off = compute_features(
            tr_tok_raw,
            tr_mask_raw,
            feature_mode=str(args_ref.feature_mode),
            feature_preprocessing=str(args_ref.feature_preprocessing),
        )
        idx_all = np.arange(len(tr_y))
        mean, std = get_mean_std(tr_feat_off, tr_mask_raw, idx_all)
        va_feat_hlt = standardize(va_feat_hlt, va_hlt_mask_raw, mean, std)
        te_feat_hlt = standardize(te_feat_hlt, te_hlt_mask_raw, mean, std)
        standardization_mode = "learned_train_split"

    va_hlt_const4 = va_hlt_tok_raw[:, :, :4].astype(np.float32)
    te_hlt_const4 = te_hlt_tok_raw[:, :, :4].astype(np.float32)
    return (
        class_names,
        va_feat_hlt.astype(np.float32),
        va_hlt_mask_raw,
        va_hlt_const4,
        va_y.astype(np.int64),
        te_feat_hlt.astype(np.float32),
        te_hlt_mask_raw,
        te_hlt_const4,
        te_y.astype(np.int64),
        standardization_mode,
    )


def _build_reconstructor_with_attrs(
    run_args: SimpleNamespace,
    input_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    reco_cfg = copy.deepcopy(BASE_RECO_CONFIG)
    reco_defaults = reco_cfg["reconstructor_model"]
    reco_cfg["reconstructor_model"]["embed_dim"] = int(_get_attr(run_args, "reco_embed_dim", reco_defaults["embed_dim"]))
    reco_cfg["reconstructor_model"]["num_heads"] = int(_get_attr(run_args, "reco_num_heads", reco_defaults["num_heads"]))
    reco_cfg["reconstructor_model"]["num_layers"] = int(_get_attr(run_args, "reco_num_layers", reco_defaults["num_layers"]))
    reco_cfg["reconstructor_model"]["ff_dim"] = int(_get_attr(run_args, "reco_ff_dim", reco_defaults["ff_dim"]))
    reco_cfg["reconstructor_model"]["dropout"] = float(_get_attr(run_args, "reco_dropout", reco_defaults["dropout"]))
    reco_cfg["reconstructor_model"]["max_split_children"] = int(
        _get_attr(run_args, "reco_max_split_children", reco_defaults["max_split_children"])
    )
    reco_cfg["reconstructor_model"]["max_generated_tokens"] = int(
        _get_attr(run_args, "reco_max_generated_tokens", reco_defaults["max_generated_tokens"])
    )

    base_reco = confgen_ops.OfflineReconstructorConfidenceHybridOps(
        input_dim=int(input_dim),
        **reco_cfg["reconstructor_model"],
    ).to(device)
    reco = v2.ReconstructorWithAttrHeads(
        base=base_reco,
        input_dim=int(input_dim),
        hidden_dim=int(_get_attr(run_args, "v2_attr_hidden_dim", 128)),
        attr_slots=int(_get_attr(run_args, "v2_attr_slots", 2)),
    ).to(device)
    return reco


def _build_dual_model(
    run_args: SimpleNamespace,
    input_dim: int,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    dual = v2.JetClassDualViewTransformer(
        input_dim_a=int(input_dim),
        input_dim_b=10,
        n_classes=int(n_classes),
        embed_dim=int(run_args.embed_dim),
        num_heads=int(run_args.num_heads),
        num_layers=int(run_args.num_layers),
        ff_dim=int(run_args.ff_dim),
        dropout=float(run_args.dropout),
    ).to(device)
    return dual


def _build_single_view_model(
    run_args: SimpleNamespace,
    input_dim: int,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    m = JetClassTransformer(
        input_dim=int(input_dim),
        n_classes=int(n_classes),
        embed_dim=int(run_args.embed_dim),
        num_heads=int(run_args.num_heads),
        num_layers=int(run_args.num_layers),
        ff_dim=int(run_args.ff_dim),
        dropout=float(run_args.dropout),
    ).to(device)
    return m


def _load_model_for_spec(
    spec: ModelSpec,
    run_args: SimpleNamespace,
    input_dim: int,
    n_classes: int,
    device: torch.device,
    corrected_weight_floor: float,
) -> LoadedModel:
    run_dir = spec.run_dir
    if spec.kind == "baseline_hlt":
        clf = _build_single_view_model(run_args, input_dim, n_classes, device)
        sd, src = _load_state_dict_from_run_dir(
            run_dir,
            ("baseline_hlt_best.pt", "baseline_best.pt", "baseline.pt"),
        )
        clf.load_state_dict(sd, strict=True)
        clf.eval()

        def _predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            x = batch["feat_hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            return clf(x, m)

        return LoadedModel(
            name=spec.name,
            kind=spec.kind,
            run_dir=run_dir,
            predict_logits=_predict,
            sources={"model_ckpt": str(src)},
            args=run_args,
        )

    if spec.kind in {"stage2", "joint", "reco_only_stagea"}:
        reco = _build_reconstructor_with_attrs(run_args, input_dim, device)
        reco_sd, reco_src = _load_state_dict_from_run_dir(
            run_dir,
            ("offline_reconstructor_stage2.pt", "offline_reconstructor.pt"),
        )
        reco.load_state_dict(reco_sd, strict=True)
        reco.eval()

        if spec.kind == "reco_only_stagea":
            clf = _build_single_view_model(run_args, 10, n_classes, device)
            clf_sd, clf_src = _load_state_dict_from_run_dir(
                run_dir,
                ("reco_only_corrected_stageA_best.pt", "reco_only_corrected_stageA.pt"),
            )
            clf.load_state_dict(clf_sd, strict=True)
            clf.eval()

            def _predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
                x = batch["feat_hlt"].to(device)
                m = batch["mask_hlt"].to(device)
                c4 = batch["const_hlt4"].to(device)
                reco_out = reco(x, m, c4, stage_scale=1.0)
                feat_b, mask_b = confgen_ops.build_soft_corrected_view_confgen_ops(
                    reco_out,
                    weight_floor=float(corrected_weight_floor),
                    scale_features_by_weight=True,
                    include_flags=False,
                )
                return clf(feat_b, mask_b)

            return LoadedModel(
                name=spec.name,
                kind=spec.kind,
                run_dir=run_dir,
                predict_logits=_predict,
                sources={"reco_ckpt": str(reco_src), "classifier_ckpt": str(clf_src)},
                args=run_args,
            )

        dual = _build_dual_model(run_args, input_dim, n_classes, device)
        if spec.kind == "stage2":
            dual_names = ("dual_joint_stage2.pt", "dual_joint.pt")
        else:
            dual_names = ("dual_joint.pt", "dual_joint_stage2.pt")
        dual_sd, dual_src = _load_state_dict_from_run_dir(run_dir, dual_names)
        dual.load_state_dict(dual_sd, strict=True)
        dual.eval()

        def _predict(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            x = batch["feat_hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            c4 = batch["const_hlt4"].to(device)
            reco_out = reco(x, m, c4, stage_scale=1.0)
            feat_b, mask_b = confgen_ops.build_soft_corrected_view_confgen_ops(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            return dual(x, m, feat_b, mask_b)

        return LoadedModel(
            name=spec.name,
            kind=spec.kind,
            run_dir=run_dir,
            predict_logits=_predict,
            sources={"reco_ckpt": str(reco_src), "dual_ckpt": str(dual_src)},
            args=run_args,
        )

    raise ValueError(f"Unsupported model kind `{spec.kind}`")


@torch.no_grad()
def _collect_logits(
    loaded: LoadedModel,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for batch in loader:
        logits = loaded.predict_logits(batch)
        y = batch["label"]
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(z)
    den = np.clip(np.sum(ez, axis=1, keepdims=True), 1e-12, np.inf)
    return ez / den


def _fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 80) -> Tuple[float, float, float]:
    device = torch.device("cpu")
    lt = torch.tensor(logits, dtype=torch.float32, device=device)
    yt = torch.tensor(y.astype(np.int64), dtype=torch.long, device=device)
    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    with torch.no_grad():
        nll_before = float(F.cross_entropy(lt, yt).item())

    def _closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3, max=1e3)
        loss = F.cross_entropy(lt / t, yt)
        loss.backward()
        return loss

    opt.step(_closure)
    t = float(torch.exp(log_t.detach()).clamp(min=1e-3, max=1e3).item())
    with torch.no_grad():
        nll_after = float(F.cross_entropy(lt / t, yt).item())
    return t, nll_before, nll_after


def _acc_from_probs(y: np.ndarray, p: np.ndarray) -> float:
    pred = np.argmax(p, axis=1)
    return float(np.mean(pred == y))


def _nll_from_probs(y: np.ndarray, p: np.ndarray) -> float:
    idx = np.arange(y.shape[0], dtype=np.int64)
    py = np.clip(p[idx, y], 1e-12, 1.0)
    return float(-np.mean(np.log(py)))


def _objective_value(y: np.ndarray, p: np.ndarray, m: Dict[str, float], optimize_for: str) -> float:
    if optimize_for == "acc":
        return float(_acc_from_probs(y, p))
    if optimize_for == "auc_macro":
        v = float(m.get("auc_macro_ovr", float("nan")))
        return v if np.isfinite(v) else float("-inf")
    if optimize_for == "sigbg_fpr50":
        v = float(m.get("signal_vs_bg_fpr50", float("nan")))
        return -v if np.isfinite(v) else float("-inf")
    if optimize_for == "targetbg_fpr50":
        v = float(m.get("target_vs_bg_ratio_fpr50", float("nan")))
        return -v if np.isfinite(v) else float("-inf")
    raise ValueError(f"Unsupported optimize_for `{optimize_for}`")


def _simplex_grid(n_models: int, step: float) -> np.ndarray:
    if n_models < 2:
        raise ValueError("Need at least 2 models for weight search.")
    k = int(round(1.0 / float(step)))
    if k <= 0:
        raise ValueError(f"Invalid step `{step}`")
    out: List[List[int]] = []

    def rec(i: int, rem: int, cur: List[int]) -> None:
        if i == n_models - 1:
            out.append(cur + [rem])
            return
        for v in range(rem + 1):
            rec(i + 1, rem - v, cur + [v])

    rec(0, k, [])
    arr = np.asarray(out, dtype=np.float64) / float(k)
    return arr


def _n_simplex_grid_candidates(n_models: int, step: float) -> int:
    if n_models < 2:
        raise ValueError("Need at least 2 models for weight search.")
    k = int(round(1.0 / float(step)))
    if k <= 0:
        raise ValueError(f"Invalid step `{step}`")
    return int(math.comb(k + n_models - 1, n_models - 1))


def _sample_simplex_dirichlet(n_models: int, n_samples: int, seed: int) -> np.ndarray:
    if n_models < 2:
        raise ValueError("Need at least 2 models for weight search.")
    n = int(max(1, n_samples))
    rng = np.random.default_rng(int(seed))
    sampled = rng.dirichlet(alpha=np.ones((n_models,), dtype=np.float64), size=n)
    # Include deterministic anchors to avoid missing edge solutions.
    uniform = np.full((1, n_models), 1.0 / float(n_models), dtype=np.float64)
    onehots = np.eye(n_models, dtype=np.float64)
    return np.concatenate([uniform, onehots, sampled], axis=0)


def _build_weight_candidates(
    n_models: int,
    step: float,
    mode: str,
    max_candidates: int,
    random_samples: int,
    random_seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    m = str(mode).lower().strip()
    if m not in {"auto", "grid", "dirichlet"}:
        raise ValueError(f"Unsupported weight search mode `{mode}`")

    grid_count = _n_simplex_grid_candidates(n_models=n_models, step=step)
    max_c = int(max(1, max_candidates))
    info: Dict[str, object] = {
        "requested_mode": m,
        "grid_candidate_count": int(grid_count),
        "max_weight_candidates": int(max_c),
    }

    if m == "grid":
        if grid_count > max_c:
            raise ValueError(
                f"Grid search would generate {grid_count} candidates "
                f"(n_models={n_models}, step={step}), exceeding --max_weight_candidates={max_c}. "
                "Use --weight_search_mode dirichlet, increase step, or raise max_weight_candidates."
            )
        w = _simplex_grid(n_models=n_models, step=step)
        info.update({"strategy": "grid", "actual_candidate_count": int(w.shape[0])})
        return w, info

    if m == "dirichlet":
        w = _sample_simplex_dirichlet(
            n_models=n_models,
            n_samples=int(max(1, random_samples)),
            seed=int(random_seed),
        )
        info.update(
            {
                "strategy": "dirichlet",
                "actual_candidate_count": int(w.shape[0]),
                "dirichlet_samples": int(max(1, random_samples)),
                "dirichlet_seed": int(random_seed),
            }
        )
        return w, info

    if grid_count <= max_c:
        w = _simplex_grid(n_models=n_models, step=step)
        info.update({"strategy": "grid_auto", "actual_candidate_count": int(w.shape[0])})
        return w, info

    w = _sample_simplex_dirichlet(
        n_models=n_models,
        n_samples=int(max(1, random_samples)),
        seed=int(random_seed),
    )
    info.update(
        {
            "strategy": "dirichlet_auto",
            "actual_candidate_count": int(w.shape[0]),
            "dirichlet_samples": int(max(1, random_samples)),
            "dirichlet_seed": int(random_seed),
        }
    )
    return w, info


def _to_float_dict(d: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _fuse_probs(weights: np.ndarray, probs_list: Sequence[np.ndarray]) -> np.ndarray:
    p = np.zeros_like(probs_list[0], dtype=np.float64)
    for i, w in enumerate(weights):
        p += float(w) * probs_list[i].astype(np.float64)
    return p.astype(np.float32)


def _fuse_logits(weights: np.ndarray, logits_list: Sequence[np.ndarray]) -> np.ndarray:
    z = np.zeros_like(logits_list[0], dtype=np.float64)
    for i, w in enumerate(weights):
        z += float(w) * logits_list[i].astype(np.float64)
    return z.astype(np.float32)


def _search_best_weights(
    weights_grid: np.ndarray,
    y_val: np.ndarray,
    probs_val_list: Sequence[np.ndarray],
    logits_val_list: Sequence[np.ndarray],
    class_names: Sequence[str],
    background_class: str,
    target_class: str,
    optimize_for: str,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    best_w = weights_grid[0].copy()
    best_score = float("-inf")
    best_tie = float("inf")
    best_metrics: Dict[str, float] = {}
    for w in weights_grid:
        if mode == "prob":
            p = _fuse_probs(w, probs_val_list)
        elif mode == "logit":
            z = _fuse_logits(w, logits_val_list)
            p = _softmax_np(z)
        else:
            raise ValueError(f"Unsupported mode `{mode}`")

        if optimize_for == "acc":
            score = _acc_from_probs(y_val, p)
            tie = _nll_from_probs(y_val, p)
            metrics = {"acc": float(score), "nll": float(tie)}
        else:
            m = eval_metrics(y_val, p, class_names, background_class, target_class)
            score = _objective_value(y_val, p, m, optimize_for)
            tie = -float(m.get("auc_macro_ovr", float("-inf")))
            metrics = _to_float_dict(m)

        if (score > best_score) or (np.isclose(score, best_score) and tie < best_tie):
            best_score = float(score)
            best_tie = float(tie)
            best_w = w.copy()
            best_metrics = metrics

    return best_w.astype(np.float32), best_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Four-model JetClass stacked fusion")
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: name:kind:run_dir (kind in baseline_hlt, stage2, joint, reco_only_stagea)",
    )
    ap.add_argument("--data_dir", type=Path, default=Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--weight_step", type=float, default=0.05, help="Simplex grid step for weighted averages.")
    ap.add_argument(
        "--weight_search_mode",
        type=str,
        default="auto",
        choices=["auto", "grid", "dirichlet"],
        help="Weight search strategy: exact grid, sampled dirichlet, or auto fallback.",
    )
    ap.add_argument(
        "--max_weight_candidates",
        type=int,
        default=200000,
        help="Maximum exact grid candidates allowed before refusing/falling back.",
    )
    ap.add_argument(
        "--weight_random_samples",
        type=int,
        default=2000,
        help="Dirichlet samples when using sampled search mode.",
    )
    ap.add_argument("--weight_random_seed", type=int, default=52)
    ap.add_argument(
        "--optimize_for",
        type=str,
        default="acc",
        choices=["acc", "auc_macro", "sigbg_fpr50", "targetbg_fpr50"],
    )
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--disable_temperature_calibration", action="store_true", default=False)
    ap.add_argument(
        "--stack_features",
        type=str,
        default="logits_probs",
        choices=["logits", "probs", "logits_probs"],
    )
    ap.add_argument(
        "--stack_Cs",
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    )
    ap.add_argument("--stack_cv", type=int, default=5)
    ap.add_argument("--stack_max_iter", type=int, default=2000)
    ap.add_argument("--stack_n_jobs", type=int, default=-1)
    args = ap.parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import JetClass dual-view dependencies. "
            "Activate the training environment (e.g. atlas_kd) with required packages "
            "(notably h5py) and rerun."
        ) from _IMPORT_ERROR
    if _EVAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import JetClass eval/data utilities. "
            "Activate the training environment with required packages "
            "(notably awkward/uproot) and rerun."
        ) from _EVAL_IMPORT_ERROR

    specs = [_parse_model_spec(s) for s in args.model]
    if len(specs) < 2:
        raise ValueError("Need at least two --model entries.")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model names detected: {names}")

    for s in specs:
        if not s.run_dir.exists():
            raise FileNotFoundError(f"Model run_dir not found: {s.run_dir}")
        if not (s.run_dir / "args.json").exists():
            raise FileNotFoundError(f"Missing args.json in run_dir: {s.run_dir}")

    run_args_map = {s.name: _ns_from_json(s.run_dir / "args.json") for s in specs}
    ref_args = run_args_map[specs[0].name]
    for s in specs[1:]:
        _check_run_compat(ref_args, run_args_map[s.name], s.name)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")

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
        standardization_mode,
    ) = _build_eval_data(ref_args, args.data_dir)

    n_classes = int(len(class_names))
    input_dim = int(va_feat_hlt.shape[-1])
    bg_name = str(ref_args.background_class)
    tgt_name = str(ref_args.target_class)

    dl_va = DataLoader(
        JointEvalDataset(va_feat_hlt, va_hlt_mask, va_hlt_const4, va_y),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    dl_te = DataLoader(
        JointEvalDataset(te_feat_hlt, te_hlt_mask, te_hlt_const4, te_y),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    loaded_models: List[LoadedModel] = []
    for s in specs:
        loaded = _load_model_for_spec(
            spec=s,
            run_args=run_args_map[s.name],
            input_dim=input_dim,
            n_classes=n_classes,
            device=device,
            corrected_weight_floor=float(args.corrected_weight_floor),
        )
        loaded_models.append(loaded)

    logits_val_list: List[np.ndarray] = []
    logits_test_list: List[np.ndarray] = []
    model_reports: Dict[str, Dict[str, object]] = {}
    yv_ref = None
    yt_ref = None
    for lm in loaded_models:
        lv, yv = _collect_logits(lm, dl_va)
        lt, yt = _collect_logits(lm, dl_te)
        if yv_ref is None:
            yv_ref = yv
            yt_ref = yt
        else:
            if not np.array_equal(yv_ref, yv):
                raise RuntimeError(f"Val label mismatch for model `{lm.name}`.")
            if not np.array_equal(yt_ref, yt):
                raise RuntimeError(f"Test label mismatch for model `{lm.name}`.")
        logits_val_list.append(lv.astype(np.float32))
        logits_test_list.append(lt.astype(np.float32))

    y_val = yv_ref.astype(np.int64)
    y_test = yt_ref.astype(np.int64)

    temps: Dict[str, float] = {}
    calibration_report: Dict[str, Dict[str, float]] = {}
    logits_val_cal: List[np.ndarray] = []
    logits_test_cal: List[np.ndarray] = []
    for lm, lv, lt in zip(loaded_models, logits_val_list, logits_test_list):
        if bool(args.disable_temperature_calibration):
            t = 1.0
            nll_before = _nll_from_probs(y_val, _softmax_np(lv))
            nll_after = nll_before
        else:
            t, nll_before, nll_after = _fit_temperature(lv, y_val, max_iter=80)
        temps[lm.name] = float(t)
        calibration_report[lm.name] = {
            "temperature": float(t),
            "val_nll_before": float(nll_before),
            "val_nll_after": float(nll_after),
        }
        logits_val_cal.append((lv / float(t)).astype(np.float32))
        logits_test_cal.append((lt / float(t)).astype(np.float32))

    probs_val_cal = [_softmax_np(z) for z in logits_val_cal]
    probs_test_cal = [_softmax_np(z) for z in logits_test_cal]

    individual_metrics: Dict[str, Dict[str, object]] = {}
    for lm, pv, pt in zip(loaded_models, probs_val_cal, probs_test_cal):
        m_val = _to_float_dict(eval_metrics(y_val, pv, class_names, bg_name, tgt_name))
        m_test = _to_float_dict(eval_metrics(y_test, pt, class_names, bg_name, tgt_name))
        individual_metrics[lm.name] = {
            "kind": lm.kind,
            "run_dir": str(lm.run_dir),
            "val": m_val,
            "test": m_test,
        }

    n_models = len(loaded_models)
    weights_uniform = np.full((n_models,), 1.0 / float(n_models), dtype=np.float32)
    p_uni_val = _fuse_probs(weights_uniform, probs_val_cal)
    p_uni_test = _fuse_probs(weights_uniform, probs_test_cal)

    weights_grid, weight_search_info = _build_weight_candidates(
        n_models=n_models,
        step=float(args.weight_step),
        mode=str(args.weight_search_mode),
        max_candidates=int(args.max_weight_candidates),
        random_samples=int(args.weight_random_samples),
        random_seed=int(args.weight_random_seed),
    )

    w_prob, w_prob_search_info = _search_best_weights(
        weights_grid=weights_grid,
        y_val=y_val,
        probs_val_list=probs_val_cal,
        logits_val_list=logits_val_cal,
        class_names=class_names,
        background_class=bg_name,
        target_class=tgt_name,
        optimize_for=str(args.optimize_for),
        mode="prob",
    )
    p_wprob_val = _fuse_probs(w_prob, probs_val_cal)
    p_wprob_test = _fuse_probs(w_prob, probs_test_cal)

    w_logit, w_logit_search_info = _search_best_weights(
        weights_grid=weights_grid,
        y_val=y_val,
        probs_val_list=probs_val_cal,
        logits_val_list=logits_val_cal,
        class_names=class_names,
        background_class=bg_name,
        target_class=tgt_name,
        optimize_for=str(args.optimize_for),
        mode="logit",
    )
    z_wlog_val = _fuse_logits(w_logit, logits_val_cal)
    z_wlog_test = _fuse_logits(w_logit, logits_test_cal)
    p_wlog_val = _softmax_np(z_wlog_val)
    p_wlog_test = _softmax_np(z_wlog_test)

    if str(args.stack_features) == "logits":
        x_val = np.concatenate(logits_val_cal, axis=1).astype(np.float32)
        x_test = np.concatenate(logits_test_cal, axis=1).astype(np.float32)
    elif str(args.stack_features) == "probs":
        x_val = np.concatenate(probs_val_cal, axis=1).astype(np.float32)
        x_test = np.concatenate(probs_test_cal, axis=1).astype(np.float32)
    else:
        x_val = np.concatenate(
            [np.concatenate(logits_val_cal, axis=1), np.concatenate(probs_val_cal, axis=1)],
            axis=1,
        ).astype(np.float32)
        x_test = np.concatenate(
            [np.concatenate(logits_test_cal, axis=1), np.concatenate(probs_test_cal, axis=1)],
            axis=1,
        ).astype(np.float32)

    stack_pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegressionCV(
            Cs=[float(c) for c in args.stack_Cs],
            cv=int(args.stack_cv),
            multi_class="multinomial",
            solver="lbfgs",
            scoring="accuracy",
            max_iter=int(args.stack_max_iter),
            n_jobs=int(args.stack_n_jobs),
            refit=True,
        ),
    )
    stack_pipe.fit(x_val, y_val)
    p_stack_val = stack_pipe.predict_proba(x_val).astype(np.float32)
    p_stack_test = stack_pipe.predict_proba(x_test).astype(np.float32)
    lr_cv = stack_pipe.named_steps["logisticregressioncv"]
    if lr_cv.C_.size > 0:
        best_c = float(np.mean(lr_cv.C_))
    else:
        best_c = float("nan")

    method_probs_val = {
        "uniform_prob_avg": p_uni_val,
        "weighted_prob_avg": p_wprob_val,
        "weighted_logit_avg": p_wlog_val,
        "stacked_logreg": p_stack_val,
    }
    method_probs_test = {
        "uniform_prob_avg": p_uni_test,
        "weighted_prob_avg": p_wprob_test,
        "weighted_logit_avg": p_wlog_test,
        "stacked_logreg": p_stack_test,
    }

    method_metrics: Dict[str, Dict[str, object]] = {}
    best_method = None
    best_score = float("-inf")
    for k in method_probs_val:
        mv = _to_float_dict(eval_metrics(y_val, method_probs_val[k], class_names, bg_name, tgt_name))
        mt = _to_float_dict(eval_metrics(y_test, method_probs_test[k], class_names, bg_name, tgt_name))
        score = _objective_value(y_val, method_probs_val[k], mv, str(args.optimize_for))
        method_metrics[k] = {"val": mv, "test": mt, "val_objective": float(score)}
        if score > best_score:
            best_score = float(score)
            best_method = k

    report = {
        "models": [
            {
                "name": lm.name,
                "kind": lm.kind,
                "run_dir": str(lm.run_dir),
                "sources": lm.sources,
            }
            for lm in loaded_models
        ],
        "data_dir": str(args.data_dir.resolve()),
        "out_dir": str(out_dir),
        "class_names": list(class_names),
        "target_class": str(tgt_name),
        "background_class": str(bg_name),
        "standardization_mode": standardization_mode,
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "optimize_for": str(args.optimize_for),
        "weight_step": float(args.weight_step),
        "weight_search_mode": str(args.weight_search_mode),
        "n_weight_candidates": int(weights_grid.shape[0]),
        "weight_search_info": weight_search_info,
        "stack_features": str(args.stack_features),
        "stack_Cs": [float(c) for c in args.stack_Cs],
        "stack_cv": int(args.stack_cv),
        "stack_best_C_mean": float(best_c),
        "temperature_calibrated": bool(not args.disable_temperature_calibration),
        "temperature_per_model": temps,
        "calibration_report": calibration_report,
        "individual_metrics": individual_metrics,
        "method_metrics": method_metrics,
        "weighted_prob_weights": {lm.name: float(w_prob[i]) for i, lm in enumerate(loaded_models)},
        "weighted_logit_weights": {lm.name: float(w_logit[i]) for i, lm in enumerate(loaded_models)},
        "weighted_prob_search_info": w_prob_search_info,
        "weighted_logit_search_info": w_logit_search_info,
        "best_method_by_val_objective": str(best_method),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    npz_payload: Dict[str, np.ndarray] = {
        "y_val": y_val.astype(np.int64),
        "y_test": y_test.astype(np.int64),
    }
    for lm, zv, zt, pv, pt in zip(loaded_models, logits_val_cal, logits_test_cal, probs_val_cal, probs_test_cal):
        safe = lm.name.replace(" ", "_")
        npz_payload[f"logits_val_{safe}"] = zv.astype(np.float32)
        npz_payload[f"logits_test_{safe}"] = zt.astype(np.float32)
        npz_payload[f"probs_val_{safe}"] = pv.astype(np.float32)
        npz_payload[f"probs_test_{safe}"] = pt.astype(np.float32)
    for k in method_probs_val:
        npz_payload[f"probs_val_{k}"] = method_probs_val[k].astype(np.float32)
        npz_payload[f"probs_test_{k}"] = method_probs_test[k].astype(np.float32)
    np.savez_compressed(out_dir / "fusion_scores.npz", **npz_payload)

    print("============================================================")
    print("JetClass Four-Model Stacked Fusion")
    print("============================================================")
    print(f"Out dir: {out_dir}")
    print(f"Models ({len(loaded_models)}):")
    for lm in loaded_models:
        print(f"  - {lm.name:16s} kind={lm.kind:16s} run={lm.run_dir}")
    print(
        f"Objective: {args.optimize_for} | weight_step={args.weight_step} | "
        f"weight_search={weight_search_info.get('strategy', 'unknown')} | "
        f"weight_candidates={weights_grid.shape[0]}"
    )
    print("------------------------------------------------------------")
    for name, info in method_metrics.items():
        mv = info["val"]
        mt = info["test"]
        print(
            f"{name:20s} "
            f"val(acc/auc/fpr50)={float(mv.get('acc', float('nan'))):.4f}/"
            f"{float(mv.get('auc_macro_ovr', float('nan'))):.4f}/"
            f"{float(mv.get('signal_vs_bg_fpr50', float('nan'))):.6f} | "
            f"test(acc/auc/fpr50)={float(mt.get('acc', float('nan'))):.4f}/"
            f"{float(mt.get('auc_macro_ovr', float('nan'))):.4f}/"
            f"{float(mt.get('signal_vs_bg_fpr50', float('nan'))):.6f}"
        )
    print("------------------------------------------------------------")
    print(f"Best method by val objective: {best_method}")
    print(f"Weighted prob weights:  {report['weighted_prob_weights']}")
    print(f"Weighted logit weights: {report['weighted_logit_weights']}")
    print(f"Saved report: {out_dir / 'report.json'}")
    print(f"Saved scores: {out_dir / 'fusion_scores.npz'}")


if __name__ == "__main__":
    main()
