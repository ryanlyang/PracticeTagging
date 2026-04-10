#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import analyze_hlt_joint31_specialization_atlas as atlas
import analyze_m2_router_signal_sweep as router
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_rhosplit_splitagain_teacherkd as stage_base
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as m2base
import reco_teacher_stageA_residual_hlt as stage_residual
import train_m9_dualreco_dualview_offdrop as dual_offdrop
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
)
from unmerge_correct_hlt import (
    DualViewCrossAttnClassifier,
    ParticleTransformer,
    compute_features,
    load_raw_constituents_from_h5,
    standardize,
)


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _load_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _pick_existing(run_dir: Path, names: List[str]) -> Path:
    for n in names:
        p = run_dir / n
        if p.exists():
            return p
    raise FileNotFoundError(f"None of candidate files found in {run_dir}: {names}")


def _infer_single_input_dim(sd: Dict[str, torch.Tensor]) -> int:
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith("input_proj.0.weight") or k.endswith("input_proj.weight"):
            return int(v.shape[1])
    raise RuntimeError("Could not infer single-view input dim from checkpoint keys")


def _infer_dual_input_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    da = None
    db = None
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith("input_proj_a.0.weight") or k.endswith("input_proj_a.weight"):
            da = int(v.shape[1])
        if k.endswith("input_proj_b.0.weight") or k.endswith("input_proj_b.weight"):
            db = int(v.shape[1])
    if da is None or db is None:
        raise RuntimeError("Could not infer dual-view input dims from checkpoint keys")
    return da, db


def _infer_optional_input_dim(sd: Dict[str, torch.Tensor], branch: str) -> int | None:
    key1 = f"input_proj_{branch}.0.weight"
    key2 = f"input_proj_{branch}.weight"
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(key1) or k.endswith(key2):
            return int(v.shape[1])
    return None


def _safe_load_state_dict(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    model_name: str,
    run_dir: Path,
    kind: str,
) -> str:
    try:
        model.load_state_dict(state, strict=True)
        return "strict"
    except RuntimeError as e:
        model_sd = model.state_dict()
        keep: Dict[str, torch.Tensor] = {}
        dropped = 0
        for k, v in state.items():
            t = model_sd.get(k, None)
            if isinstance(v, torch.Tensor) and isinstance(t, torch.Tensor) and tuple(v.shape) == tuple(t.shape):
                keep[k] = v
            else:
                dropped += 1
        info = model.load_state_dict(keep, strict=False)
        print(
            f"[warn] {kind} compat-load for model={model_name} run_dir={run_dir} "
            f"because {_short_err(e)} | kept={len(keep)} dropped={dropped} "
            f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}"
        )
        return (
            f"compat(missing={len(info.missing_keys)},unexpected={len(info.unexpected_keys)},"
            f"kept={len(keep)},dropped={dropped})"
        )


def _iter_batches(n: int, batch_size: int):
    for s in range(0, int(n), int(batch_size)):
        e = min(int(n), s + int(batch_size))
        yield s, e


def _load_means_stds(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    p = run_dir / "data_splits.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing data_splits.npz: {p}")
    z = np.load(p, allow_pickle=False)
    if "means" in z and "stds" in z:
        return np.asarray(z["means"], dtype=np.float32), np.asarray(z["stds"], dtype=np.float32)
    # Older concat-teacher runs saved separate normalization tracks.
    if "means_off" in z and "stds_off" in z:
        return np.asarray(z["means_off"], dtype=np.float32), np.asarray(z["stds_off"], dtype=np.float32)
    if "means_hlt" in z and "stds_hlt" in z:
        return np.asarray(z["means_hlt"], dtype=np.float32), np.asarray(z["stds_hlt"], dtype=np.float32)
    raise KeyError(f"data_splits missing usable means/stds keys: {p} (keys={list(z.keys())})")


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    yb = np.asarray(y, dtype=np.int64)
    pp = np.asarray(p, dtype=np.float64)
    if yb.size == 0 or np.unique(yb).size < 2:
        return float("nan")
    return float(roc_auc_score(yb, pp))


def _short_err(exc: Exception, max_len: int = 220) -> str:
    s = str(exc).replace("\n", " ")
    if len(s) <= int(max_len):
        return s
    return s[: int(max_len) - 3] + "..."


_UNMERGE_WRAPPER_PREFIX = "offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly"


def _import_or_reload(mod_name: str):
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name]), "reload"
    return importlib.import_module(mod_name), "import"


def _safe_model_filename(model_name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("._")
    return s or "model"


def _model_cache_path(cache_dir: Path, idx: int, model_name: str) -> Path:
    return cache_dir / f"{int(idx)+1:03d}_{_safe_model_filename(model_name)}.npy"


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.asarray(arr, dtype=np.float32))
    tmp.replace(path)


def _variant_suffixes_from_run_name(run_name: str) -> List[str]:
    r = str(run_name)
    r = re.sub(r"_seed\d+$", "", r)
    r = re.sub(r"_(\d+[kKmM]\d+[kKmM]\d+[kKmM])$", "", r)
    markers = [
        "fulltrain_prog_unfreeze_",
        "stagec_prog_unfreeze_",
        "stagec_prog_",
        "delta000_",
        "delta005_",
    ]
    out: List[str] = []
    for mk in markers:
        if mk in r:
            suf = r.split(mk, 1)[1].strip("_")
            if suf:
                out.append(suf)
    # preserve order, remove dups
    uniq: List[str] = []
    seen = set()
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _activate_joint_wrapper_for_run(run_dir: Path, model_name: str) -> str:
    run_name = str(run_dir.name)
    suffixes = _variant_suffixes_from_run_name(run_name)
    candidates = [f"{_UNMERGE_WRAPPER_PREFIX}_{s}" for s in suffixes]
    for mod_name in candidates:
        try:
            # Important: wrappers monkeypatch m2base on import. Since we reload m2base per model,
            # we need wrapper re-execution only when wrapper was already loaded.
            mod, mode = _import_or_reload(mod_name)
            # Some wrappers (e.g. jetlatent_set2set) only patch inside an explicit hook.
            patch_fn = getattr(mod, "_patch_base_module", None)
            if callable(patch_fn):
                patch_fn()
            return f"{mod_name}({mode})"
        except ModuleNotFoundError as e:
            # Only ignore if the missing module is exactly the candidate wrapper module.
            if getattr(e, "name", "") == mod_name:
                continue
            print(
                f"[warn] wrapper import failed for model={model_name} run={run_name} "
                f"module={mod_name}: {_short_err(e)}"
            )
        except Exception as e:
            print(
                f"[warn] wrapper import failed for model={model_name} run={run_name} "
                f"module={mod_name}: {_short_err(e)}"
            )
    return "base"


def _looks_like_set2set_checkpoint(reco_sd: Dict[str, torch.Tensor]) -> bool:
    aw = reco_sd.get("action_head.weight")
    sew = reco_sd.get("split_exist_head.weight")
    sdw = reco_sd.get("split_delta_head.weight")
    if not (isinstance(aw, torch.Tensor) and isinstance(sew, torch.Tensor) and isinstance(sdw, torch.Tensor)):
        return False
    if aw.ndim != 2 or sew.ndim != 2 or sdw.ndim != 2:
        return False
    return bool(
        int(aw.shape[0]) == 1
        and int(sew.shape[0]) == 1
        and int(sdw.shape[0]) == int(sdw.shape[1])
        and int(sdw.shape[0]) >= 64
    )


def _family_from_score_file(model: str, score_path: Path) -> str:
    if model == "hlt":
        return "hlt"
    name = score_path.name
    if name == "fusion_scores_val_test.npz":
        return "joint"
    if name == "stageA_only_scores.npz":
        if model == "reco_teacher_s09":
            return "stagea_reco_teacher"
        return "stagea_corrected_only"
    if name == "concat_teacher_stageA_scores.npz":
        return "stagea_corrected_only"
    if name == "stageA_residual_scores.npz":
        return "stagea_residual"
    if name == "dualreco_dualview_scores.npz":
        return "dualreco_dualview"
    raise RuntimeError(f"Unknown score-file family for model={model}: {score_path}")


def _feature_ablation_mode_from_model_name(model: str) -> str:
    m = str(model).lower()
    if "noangle" in m:
        return "no_angle"
    if "noscale" in m:
        return "no_scale"
    if "coreshape" in m:
        return "core_shape"
    return "none"


def _infer_hlt_baseline(
    run_dir: Path,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: Dict,
) -> np.ndarray:
    sd = _load_state(_pick_existing(run_dir, ["baseline.pt"]), device)
    in_dim = _infer_single_input_dim(sd)
    model = ParticleTransformer(input_dim=int(in_dim), **cfg["model"]).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
    with torch.no_grad():
        for s, e in _iter_batches(n, batch_size):
            x_np = standardize(feat_hlt[s:e], mask_hlt[s:e], means, stds).astype(np.float32)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            logit = model(x, m).squeeze(1)
            out[s:e] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return out


def _infer_joint_score(
    model_name: str,
    run_dir: Path,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: Dict,
    corrected_weight_floor: float,
) -> Tuple[np.ndarray, str]:
    wrapper_used = _activate_joint_wrapper_for_run(run_dir=run_dir, model_name=model_name)
    reco_sd = _load_state(
        _pick_existing(
            run_dir,
            [
                "offline_reconstructor.pt",
                "offline_reconstructor_bestfpr50.pt",
                "offline_reconstructor_stage2.pt",
                "offline_reconstructor_stage2_bestfpr50.pt",
            ],
        ),
        device,
    )
    dual_sd = _load_state(
        _pick_existing(
            run_dir,
            [
                "dual_joint.pt",
                "dual_joint_bestfpr50.pt",
                "dual_joint_stage2.pt",
                "dual_joint_stage2_bestfpr50.pt",
            ],
        ),
        device,
    )

    # Safety net: some legacy set2set runs only patch the base module via an explicit hook.
    # If run-name wrapper resolution missed it, auto-activate from checkpoint signature.
    if wrapper_used == "base" and _looks_like_set2set_checkpoint(reco_sd):
        mod_name = f"{_UNMERGE_WRAPPER_PREFIX}_jetlatent_set2set"
        try:
            mod, mode = _import_or_reload(mod_name)
            patch_fn = getattr(mod, "_patch_base_module", None)
            if callable(patch_fn):
                patch_fn()
                wrapper_used = f"{mod_name}(auto:{mode})"
        except Exception as e:
            print(
                f"[warn] set2set auto-wrapper activation failed for model={model_name} "
                f"run_dir={run_dir}: {_short_err(e)}"
            )

    reco_cls = getattr(m2base, "OfflineReconstructor", OfflineReconstructor)
    reco = reco_cls(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco_loader_mode = _safe_load_state_dict(
        model=reco,
        state=reco_sd,
        model_name=model_name,
        run_dir=run_dir,
        kind="reconstructor",
    )

    da, db = _infer_dual_input_dims(dual_sd)
    dc = _infer_optional_input_dim(dual_sd, "c")
    dual_cls = getattr(m2base, "DualViewCrossAttnClassifier", DualViewCrossAttnClassifier)
    dual_kwargs = dict(cfg["model"])
    dual_kwargs.update({"input_dim_a": int(da), "input_dim_b": int(db)})
    try:
        sig = inspect.signature(dual_cls)
        if "input_dim_c" in sig.parameters and dc is not None:
            dual_kwargs["input_dim_c"] = int(dc)
    except Exception:
        pass
    try:
        dual = dual_cls(**dual_kwargs).to(device)
    except Exception as e:
        print(
            f"[warn] dual classifier ctor fallback for model={model_name} run_dir={run_dir} "
            f"wrapper={wrapper_used} because {_short_err(e)}"
        )
        dual = DualViewCrossAttnClassifier(input_dim_a=int(da), input_dim_b=int(db), **cfg["model"]).to(device)
    dual_loader_mode = _safe_load_state_dict(
        model=dual,
        state=dual_sd,
        model_name=model_name,
        run_dir=run_dir,
        kind="dual",
    )
    reco.eval()
    dual.eval()

    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
    with torch.no_grad():
        for s, e in _iter_batches(n, batch_size):
            x_np = standardize(feat_hlt[s:e], mask_hlt[s:e], means, stds).astype(np.float32)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            reco_out = reco(x, m, c, stage_scale=1.0)
            feat_b, mask_b = m2base.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            logit = dual(x, m, feat_b, mask_b).squeeze(1)
            out[s:e] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return out, f"{wrapper_used}|reco:{reco_loader_mode}|dual:{dual_loader_mode}"


def _infer_stagea_reco_teacher_score(
    run_dir: Path,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: Dict,
    corrected_weight_floor: float,
) -> np.ndarray:
    teacher_sd = _load_state(_pick_existing(run_dir, ["teacher.pt"]), device)
    reco_sd = _load_state(
        _pick_existing(run_dir, ["offline_reconstructor_stageA.pt", "offline_reconstructor.pt"]),
        device,
    )

    teacher_in = _infer_single_input_dim(teacher_sd)
    teacher = ParticleTransformer(input_dim=int(teacher_in), **cfg["model"]).to(device)
    teacher.load_state_dict(teacher_sd, strict=True)

    reco = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco.load_state_dict(reco_sd, strict=True)
    teacher.eval()
    reco.eval()

    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(np.clip(stds, 1e-6, None), dtype=torch.float32, device=device)

    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
    with torch.no_grad():
        for s, e in _iter_batches(n, batch_size):
            x_np = standardize(feat_hlt[s:e], mask_hlt[s:e], means, stds).astype(np.float32)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            reco_out = reco(x, m, c, stage_scale=1.0)
            feat_r, mask_r = stage_base._build_teacher_reco_features_from_output(
                reco_out,
                c,
                m,
                weight_floor=float(corrected_weight_floor),
            )
            feat_r_std = stage_base._standardize_features_torch(feat_r, mask_r, means_t, stds_t)
            if int(feat_r_std.shape[-1]) < int(teacher_in):
                raise RuntimeError(
                    f"Teacher input_dim={teacher_in} but reco features have dim={feat_r_std.shape[-1]} in {run_dir}"
                )
            logit = teacher(feat_r_std[:, :, : int(teacher_in)], mask_r).squeeze(1)
            out[s:e] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return out


def _infer_stagea_corrected_only_score(
    run_dir: Path,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: Dict,
    corrected_weight_floor: float,
) -> np.ndarray:
    corrected_sd = _load_state(
        _pick_existing(run_dir, ["corrected_only_tagger.pt", "corrected_only_joint_tagger.pt"]),
        device,
    )
    reco_sd = _load_state(
        _pick_existing(run_dir, ["offline_reconstructor_stageA.pt", "offline_reconstructor.pt"]),
        device,
    )

    in_dim = _infer_single_input_dim(corrected_sd)
    corrected = ParticleTransformer(input_dim=int(in_dim), **cfg["model"]).to(device)
    corrected.load_state_dict(corrected_sd, strict=True)

    reco = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco.load_state_dict(reco_sd, strict=True)
    corrected.eval()
    reco.eval()

    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
    with torch.no_grad():
        for s, e in _iter_batches(n, batch_size):
            x_np = standardize(feat_hlt[s:e], mask_hlt[s:e], means, stds).astype(np.float32)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            reco_out = reco(x, m, c, stage_scale=1.0)
            feat_c, mask_c = stage_base.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            if int(feat_c.shape[-1]) < int(in_dim):
                raise RuntimeError(
                    f"Corrected-only input_dim={in_dim} but corrected features have dim={feat_c.shape[-1]} in {run_dir}"
                )
            logit = corrected(feat_c[:, :, : int(in_dim)], mask_c).squeeze(1)
            out[s:e] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return out


def _infer_stagea_residual_score(
    run_dir: Path,
    source_score_npz: Path,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: Dict,
    corrected_weight_floor: float,
) -> np.ndarray:
    baseline_sd = _load_state(_pick_existing(run_dir, ["baseline.pt"]), device)
    reco_sd = _load_state(
        _pick_existing(run_dir, ["offline_reconstructor_stageA.pt", "offline_reconstructor.pt"]),
        device,
    )
    residual_sd = _load_state(_pick_existing(run_dir, ["residual_head.pt"]), device)

    alpha = float("nan")
    z = np.load(source_score_npz, allow_pickle=False)
    if "alpha_residual_joint" in z:
        alpha_j = float(np.asarray(z["alpha_residual_joint"]).reshape(()))
        if np.isfinite(alpha_j):
            alpha = alpha_j
    if not np.isfinite(alpha) and "alpha_residual_frozen" in z:
        alpha_f = float(np.asarray(z["alpha_residual_frozen"]).reshape(()))
        if np.isfinite(alpha_f):
            alpha = alpha_f
    if not np.isfinite(alpha):
        alpha = 1.0

    hlt_in = _infer_single_input_dim(baseline_sd)
    baseline = ParticleTransformer(input_dim=int(hlt_in), **cfg["model"]).to(device)
    baseline.load_state_dict(baseline_sd, strict=True)

    reco = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco.load_state_dict(reco_sd, strict=True)

    res_in = _infer_single_input_dim(residual_sd)
    residual = stage_residual.ResidualHead(input_dim=int(res_in), model_cfg=cfg["model"]).to(device)
    residual.load_state_dict(residual_sd, strict=True)

    baseline.eval()
    reco.eval()
    residual.eval()

    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
    with torch.no_grad():
        for s, e in _iter_batches(n, batch_size):
            x_np = standardize(feat_hlt[s:e], mask_hlt[s:e], means, stds).astype(np.float32)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            hlt_logit = baseline(x, m).squeeze(1)
            reco_out = reco(x, m, c, stage_scale=1.0)
            feat_c, mask_c = stage_base.build_soft_corrected_view(
                reco_out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            if int(feat_c.shape[-1]) < int(res_in):
                raise RuntimeError(
                    f"Residual input_dim={res_in} but corrected features have dim={feat_c.shape[-1]} in {run_dir}"
                )
            rhat = residual(feat_c[:, :, : int(res_in)], mask_c)
            score = torch.sigmoid(hlt_logit + float(alpha) * rhat)
            out[s:e] = score.detach().cpu().numpy().astype(np.float32)
    return out


def _infer_dualreco_dualview_score(
    run_dir: Path,
    model_name: str,
    feat_hlt: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    batch_size: int,
    cfg: Dict,
    corrected_weight_floor: float,
) -> np.ndarray:
    reco_a_sd = _load_state(_pick_existing(run_dir, ["offline_reconstructor_A_stageA.pt"]), device)
    reco_b_sd = _load_state(_pick_existing(run_dir, ["offline_reconstructor_B_stageA.pt"]), device)
    dual_sd = _load_state(_pick_existing(run_dir, ["dualview_frozen.pt"]), device)

    reco_a = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco_a.load_state_dict(reco_a_sd, strict=True)
    reco_b = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reco_b.load_state_dict(reco_b_sd, strict=True)

    da, db = _infer_dual_input_dims(dual_sd)
    dual = DualViewCrossAttnClassifier(input_dim_a=int(da), input_dim_b=int(db), **cfg["model"]).to(device)
    dual.load_state_dict(dual_sd, strict=True)

    feat_mode = _feature_ablation_mode_from_model_name(model_name)

    reco_a.eval()
    reco_b.eval()
    dual.eval()

    n = int(feat_hlt.shape[0])
    out = np.zeros((n,), dtype=np.float32)
    with torch.no_grad():
        for s, e in _iter_batches(n, batch_size):
            x_np = standardize(feat_hlt[s:e], mask_hlt[s:e], means, stds).astype(np.float32)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(mask_hlt[s:e]).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(const_hlt[s:e]).to(device=device, dtype=torch.float32)

            out_a = reco_a(x, m, c, stage_scale=1.0)
            feat_a, mask_a = stage_base.build_soft_corrected_view(
                out_a,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            out_b = reco_b(x, m, c, stage_scale=1.0)
            feat_b, mask_b = m2base.build_soft_corrected_view(
                out_b,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=False,
            )
            feat_b = dual_offdrop.apply_feature_ablation_to_corrected_torch(feat_b, mask_b, feat_mode)

            if int(feat_a.shape[-1]) < int(da):
                raise RuntimeError(f"Dual-A input dim mismatch in {run_dir}: have {feat_a.shape[-1]} need {da}")
            if int(feat_b.shape[-1]) < int(db):
                raise RuntimeError(f"Dual-B input dim mismatch in {run_dir}: have {feat_b.shape[-1]} need {db}")
            logit = dual(feat_a[:, :, : int(da)], mask_a, feat_b[:, :, : int(db)], mask_b).squeeze(1)
            out[s:e] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute model scores on a large dev pool for rolling greedy fusion.")
    ap.add_argument("--fusion_json", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--dev_offset_jets", type=int, default=375000)
    ap.add_argument("--dev_n_jets", type=int, default=1000000)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--hlt_seed", type=int, default=-1, help="If <0, use seed from anchor data_setup.json")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--corrected_weight_floor_joint", type=float, default=1e-4)
    ap.add_argument("--corrected_weight_floor_stagea", type=float, default=0.03)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--scores_npz", type=str, default="")
    ap.add_argument("--manifest_json", type=str, default="")
    ap.add_argument("--monitor_csv", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    fusion_json = Path(args.fusion_json).expanduser().resolve()
    if not fusion_json.exists():
        raise FileNotFoundError(f"fusion_json not found: {fusion_json}")
    fusion = json.loads(fusion_json.read_text())

    y_val_ref, y_test, _scores_val_ref, scores_test_ref, score_files_used, run_dirs, model_order = atlas._load_scores_from_fusion_json(
        fusion_obj=fusion,
        fusion_json_path=fusion_json,
    )
    del y_val_ref

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (fusion_json.parent / f"precomputed_devpool_{int(args.dev_n_jets)}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_npz = (
        Path(args.scores_npz).expanduser().resolve()
        if str(args.scores_npz).strip()
        else (out_dir / "precomputed_scores_devpool.npz")
    )
    manifest_json = (
        Path(args.manifest_json).expanduser().resolve()
        if str(args.manifest_json).strip()
        else (out_dir / "precomputed_scores_manifest.json")
    )
    monitor_csv = (
        Path(args.monitor_csv).expanduser().resolve()
        if str(args.monitor_csv).strip()
        else (out_dir / "precompute_model_monitor.csv")
    )
    model_cache_dir = out_dir / "model_score_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    anchor_run_dir = run_dirs.get("joint_delta_run_dir")
    if anchor_run_dir is None:
        p_joint = score_files_used.get("joint_delta", "")
        if not p_joint:
            raise KeyError("Could not resolve anchor run_dir from fusion json (joint_delta).")
        anchor_run_dir = Path(p_joint).expanduser().resolve().parent
    anchor_run_dir = Path(anchor_run_dir).expanduser().resolve()

    setup_path = anchor_run_dir / "data_setup.json"
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing anchor data_setup.json: {setup_path}")
    with setup_path.open("r", encoding="utf-8") as f:
        setup = json.load(f)

    cfg_anchor = _deepcopy_cfg()
    if isinstance(setup.get("hlt_effects"), dict):
        cfg_anchor["hlt_effects"].update(setup["hlt_effects"])
    if int(args.hlt_seed) >= 0:
        hlt_seed = int(args.hlt_seed)
    else:
        hlt_seed = int(setup.get("seed", 0))

    train_files = router._build_train_file_list(setup, args.train_path)
    n_need = int(args.dev_offset_jets) + int(args.dev_n_jets)
    print("Loading offline constituents...")
    all_const, all_labels = load_raw_constituents_from_h5(
        train_files,
        max_jets=n_need,
        max_constits=int(args.max_constits),
    )

    const_raw = np.asarray(
        all_const[int(args.dev_offset_jets) : int(args.dev_offset_jets) + int(args.dev_n_jets)],
        dtype=np.float32,
    )
    labels_dev = np.asarray(
        all_labels[int(args.dev_offset_jets) : int(args.dev_offset_jets) + int(args.dev_n_jets)],
        dtype=np.int64,
    )
    if const_raw.shape[0] != int(args.dev_n_jets):
        raise RuntimeError(
            f"Requested dev_n_jets={int(args.dev_n_jets)} but only loaded {const_raw.shape[0]} jets "
            f"(offset={int(args.dev_offset_jets)}, max_jets={n_need})."
        )

    print("Generating pseudo-HLT...")
    const_off, mask_off = router._offline_mask(const_raw, float(cfg_anchor["hlt_effects"]["pt_threshold_offline"]))
    hlt_const, hlt_mask, _, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        mask_off,
        cfg_anchor,
        seed=int(hlt_seed),
    )
    print("Computing HLT features...")
    feat_hlt = compute_features(hlt_const, hlt_mask).astype(np.float32)
    del all_const, all_labels, const_raw, const_off, mask_off

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Models to precompute: {len(model_order)}")
    print(f"Dev pool size: {labels_dev.shape[0]}")

    scores_dev_mat = np.zeros((len(model_order), labels_dev.shape[0]), dtype=np.float32)
    scores_test_mat = np.vstack([np.asarray(scores_test_ref[m], dtype=np.float32) for m in model_order]).astype(np.float32)
    labels_test = np.asarray(y_test, dtype=np.float32)

    monitor_rows: List[Dict[str, object]] = []
    family_by_model: Dict[str, str] = {}

    # hlt is always anchored to joint_delta baseline.
    try:
        means_anchor, stds_anchor = _load_means_stds(anchor_run_dir)
        hlt_probs_anchor = _infer_hlt_baseline(
            run_dir=anchor_run_dir,
            feat_hlt=feat_hlt,
            mask_hlt=hlt_mask,
            means=means_anchor,
            stds=stds_anchor,
            device=device,
            batch_size=int(args.batch_size),
            cfg=_deepcopy_cfg(),
        )
    except Exception:
        # If baseline inference fails, fall back to joint_delta source val scores only if sizes match.
        hlt_probs_anchor = np.zeros((labels_dev.shape[0],), dtype=np.float32)
        raise

    score_paths: Dict[str, Path] = {}
    for m in model_order:
        if m == "hlt":
            score_paths[m] = Path(score_files_used.get("joint_delta", "")).expanduser().resolve()
        else:
            p = score_files_used.get(m, "")
            if not p:
                raise KeyError(f"Missing score_file_used entry for model={m}")
            score_paths[m] = Path(p).expanduser().resolve()

    for i, model_name in enumerate(model_order):
        t_model = time.time()
        cache_path = _model_cache_path(model_cache_dir, i, model_name)
        score_dev = None
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                cached = np.asarray(cached, dtype=np.float32).reshape(-1)
                if int(cached.shape[0]) == int(labels_dev.shape[0]):
                    score_dev = cached
                else:
                    print(
                        f"[warn] cache size mismatch for model={model_name}: "
                        f"{cached.shape[0]} vs dev {labels_dev.shape[0]}; recomputing."
                    )
            except Exception as e:
                print(f"[warn] failed to load cache for model={model_name} at {cache_path}: {_short_err(e)}")

        if score_dev is not None:
            if model_name == "hlt":
                fam = "hlt"
                run_dir = anchor_run_dir
                stats_source = "anchor_hlt(cache)"
                joint_loader_meta = ""
            else:
                score_path = score_paths[model_name]
                run_dir = score_path.parent
                fam = _family_from_score_file(model_name, score_path)
                stats_source = "cache_resume"
                joint_loader_meta = "cache"

            family_by_model[model_name] = fam
            scores_dev_mat[i] = np.asarray(score_dev, dtype=np.float32)
            auc_dev = _safe_auc(labels_dev, score_dev)
            auc_test = _safe_auc(labels_test, scores_test_mat[i])
            dt = time.time() - t_model
            monitor_rows.append(
                {
                    "model": model_name,
                    "family": fam,
                    "run_dir": str(run_dir),
                    "score_file": str(score_paths.get(model_name, "")),
                    "stats_source": stats_source,
                    "joint_loader": joint_loader_meta,
                    "auc_dev": float(auc_dev),
                    "auc_test_source": float(auc_test),
                    "elapsed_sec": float(dt),
                    "cache_hit": 1,
                }
            )
            print(
                f"[{i+1:02d}/{len(model_order):02d}] {model_name:50s} "
                f"family={fam:22s} auc_dev={auc_dev:.6f} auc_test_src={auc_test:.6f} "
                f"t={dt:.1f}s [cache]"
            )
            _save_csv(monitor_csv, monitor_rows)
            continue

        # Reset any prior wrapper monkeypatches before resolving current model.
        importlib.reload(m2base)
        joint_loader_meta = ""
        if model_name == "hlt":
            fam = "hlt"
            score_dev = hlt_probs_anchor
            run_dir = anchor_run_dir
            stats_source = "anchor_hlt"
        else:
            score_path = score_paths[model_name]
            run_dir = score_path.parent
            fam = _family_from_score_file(model_name, score_path)
            stats_source = "run_dir"
            try:
                means, stds = _load_means_stds(run_dir)
            except Exception as e:
                # Some legacy runs miss per-run stats keys; fall back to anchor stats so
                # precompute continues and diagnostics can still flag weak models.
                means, stds = means_anchor, stds_anchor
                stats_source = f"anchor_fallback:{type(e).__name__}"
                print(
                    f"[warn] using anchor means/stds for model={model_name} run_dir={run_dir} "
                    f"because {_short_err(e)}"
                )
            cfg = _deepcopy_cfg()

            if fam == "joint":
                score_dev, joint_loader_meta = _infer_joint_score(
                    model_name=model_name,
                    run_dir=run_dir,
                    feat_hlt=feat_hlt,
                    mask_hlt=hlt_mask,
                    const_hlt=hlt_const,
                    means=means,
                    stds=stds,
                    device=device,
                    batch_size=int(args.batch_size),
                    cfg=cfg,
                    corrected_weight_floor=float(args.corrected_weight_floor_joint),
                )
            elif fam == "stagea_reco_teacher":
                score_dev = _infer_stagea_reco_teacher_score(
                    run_dir=run_dir,
                    feat_hlt=feat_hlt,
                    mask_hlt=hlt_mask,
                    const_hlt=hlt_const,
                    means=means,
                    stds=stds,
                    device=device,
                    batch_size=int(args.batch_size),
                    cfg=cfg,
                    corrected_weight_floor=float(args.corrected_weight_floor_stagea),
                )
            elif fam == "stagea_corrected_only":
                score_dev = _infer_stagea_corrected_only_score(
                    run_dir=run_dir,
                    feat_hlt=feat_hlt,
                    mask_hlt=hlt_mask,
                    const_hlt=hlt_const,
                    means=means,
                    stds=stds,
                    device=device,
                    batch_size=int(args.batch_size),
                    cfg=cfg,
                    corrected_weight_floor=float(args.corrected_weight_floor_stagea),
                )
            elif fam == "stagea_residual":
                score_dev = _infer_stagea_residual_score(
                    run_dir=run_dir,
                    source_score_npz=score_path,
                    feat_hlt=feat_hlt,
                    mask_hlt=hlt_mask,
                    const_hlt=hlt_const,
                    means=means,
                    stds=stds,
                    device=device,
                    batch_size=int(args.batch_size),
                    cfg=cfg,
                    corrected_weight_floor=float(args.corrected_weight_floor_stagea),
                )
            elif fam == "dualreco_dualview":
                score_dev = _infer_dualreco_dualview_score(
                    run_dir=run_dir,
                    model_name=model_name,
                    feat_hlt=feat_hlt,
                    mask_hlt=hlt_mask,
                    const_hlt=hlt_const,
                    means=means,
                    stds=stds,
                    device=device,
                    batch_size=int(args.batch_size),
                    cfg=cfg,
                    corrected_weight_floor=float(args.corrected_weight_floor_stagea),
                )
            else:
                raise RuntimeError(f"Unsupported family={fam} for model={model_name}")

        family_by_model[model_name] = fam
        scores_dev_mat[i] = np.asarray(score_dev, dtype=np.float32)
        _atomic_save_npy(cache_path, np.asarray(score_dev, dtype=np.float32))

        auc_dev = _safe_auc(labels_dev, score_dev)
        auc_test = _safe_auc(labels_test, scores_test_mat[i])
        dt = time.time() - t_model
        monitor_rows.append(
            {
                "model": model_name,
                "family": fam,
                "run_dir": str(run_dir),
                "score_file": str(score_paths.get(model_name, "")),
                "stats_source": stats_source if model_name != "hlt" else "anchor_hlt",
                "joint_loader": joint_loader_meta,
                "auc_dev": float(auc_dev),
                "auc_test_source": float(auc_test),
                "elapsed_sec": float(dt),
                "cache_hit": 0,
            }
        )
        print(
            f"[{i+1:02d}/{len(model_order):02d}] {model_name:50s} "
            f"family={fam:22s} auc_dev={auc_dev:.6f} auc_test_src={auc_test:.6f} t={dt:.1f}s"
        )
        _save_csv(monitor_csv, monitor_rows)

    np.savez_compressed(
        scores_npz,
        labels_dev=labels_dev.astype(np.float32),
        labels_test=labels_test.astype(np.float32),
        scores_dev=scores_dev_mat.astype(np.float32),
        scores_test=scores_test_mat.astype(np.float32),
        dev_offset_jets=np.asarray(int(args.dev_offset_jets), dtype=np.int64),
        dev_n_jets=np.asarray(int(args.dev_n_jets), dtype=np.int64),
    )

    _save_csv(monitor_csv, monitor_rows)

    manifest = {
        "fusion_json": str(fusion_json),
        "scores_npz": str(scores_npz),
        "model_order": list(model_order),
        "score_files_used": {k: str(v) for k, v in score_paths.items()},
        "family_by_model": family_by_model,
        "dev_pool": {
            "offset_jets": int(args.dev_offset_jets),
            "n_jets": int(args.dev_n_jets),
            "hlt_seed": int(hlt_seed),
            "max_constits": int(args.max_constits),
        },
        "settings": vars(args),
        "monitor_csv": str(monitor_csv),
        "model_cache_dir": str(model_cache_dir),
        "timing_sec": float(time.time() - t0),
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("Precompute Dev-Pool Scores (55-model ready)")
    print("=" * 72)
    print(f"Fusion json:   {fusion_json}")
    print(f"Out dir:       {out_dir}")
    print(f"Scores npz:    {scores_npz}")
    print(f"Manifest json: {manifest_json}")
    print(f"Monitor csv:   {monitor_csv}")
    print(f"Cache dir:     {model_cache_dir}")
    print(f"Models:        {len(model_order)}")
    print(f"Dev jets:      {int(args.dev_n_jets)} (offset={int(args.dev_offset_jets)})")


if __name__ == "__main__":
    main()
