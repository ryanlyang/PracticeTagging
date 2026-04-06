#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from unmerge_correct_hlt import (
    ParticleTransformer,
    DualViewCrossAttnClassifier,
    compute_features,
    get_stats,
    load_raw_constituents_from_h5,
    standardize,
)
from offline_reconstructor_no_gt_local30kv2 import (
    CONFIG as BASE_CONFIG,
    OfflineReconstructor,
    apply_hlt_effects_realistic_nomap,
)
import offline_reconstructor_joint_dualview_stage2save_auc_norankc_nopriv_unmergeonly as joint_base


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    return np.log(p / (1.0 - p))


def _entropy_bernoulli(p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _js_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    q = _clip_probs(q)
    m = 0.5 * (p + q)
    kl_pm = p * np.log(p / m) + (1.0 - p) * np.log((1.0 - p) / (1.0 - m))
    kl_qm = q * np.log(q / m) + (1.0 - q) * np.log((1.0 - q) / (1.0 - m))
    return 0.5 * (kl_pm + kl_qm)


def _bce_per_jet(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _fpr_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = _clip_probs(p)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    idx = int(np.argmin(np.abs(tpr - float(target_tpr))))
    return float(fpr[idx])


def _threshold_at_target_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = _clip_probs(p)
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= float(target_tpr))[0]
    if idx.size == 0:
        return float(thr[-1])
    return float(thr[idx[0]])


def _load_ckpt_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _infer_dual_input_dims(dual_sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "input_proj_a.0.weight" in dual_sd:
        da = int(dual_sd["input_proj_a.0.weight"].shape[1])
    elif "input_proj_a.weight" in dual_sd:
        da = int(dual_sd["input_proj_a.weight"].shape[1])
    else:
        raise RuntimeError("Could not infer dual input_dim_a from checkpoint keys")

    if "input_proj_b.0.weight" in dual_sd:
        db = int(dual_sd["input_proj_b.0.weight"].shape[1])
    elif "input_proj_b.weight" in dual_sd:
        db = int(dual_sd["input_proj_b.weight"].shape[1])
    else:
        raise RuntimeError("Could not infer dual input_dim_b from checkpoint keys")
    return da, db


def _build_train_file_list(data_setup: Dict, train_path_arg: str) -> List[Path]:
    tf = [Path(p) for p in data_setup.get("train_files", []) if str(p).strip()]
    if tf:
        return tf
    train_path = Path(train_path_arg)
    if train_path.is_dir():
        fs = sorted(train_path.glob("*.h5"))
    else:
        fs = [Path(p) for p in str(train_path_arg).split(",") if str(p).strip()]
    if len(fs) == 0:
        raise FileNotFoundError(f"No .h5 files found from --train_path={train_path_arg}")
    return fs


def _offline_mask(const_raw: np.ndarray, pt_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    raw_mask = const_raw[:, :, 0] > 0.0
    mask_off = raw_mask & (const_raw[:, :, 0] >= float(pt_thr))
    const_off = const_raw.copy()
    const_off[~mask_off] = 0.0
    return const_off.astype(np.float32), mask_off.astype(bool)


def _wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


def _merge_two_tokens(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Inputs are [pt, eta, phi, E]
    pa, ea, fa, Ea = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    pb, eb, fb, Eb = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    pxa = pa * math.cos(fa)
    pya = pa * math.sin(fa)
    pza = pa * math.sinh(ea)
    pxb = pb * math.cos(fb)
    pyb = pb * math.sin(fb)
    pzb = pb * math.sinh(eb)

    px = pxa + pxb
    py = pya + pyb
    pz = pza + pzb
    E = max(Ea + Eb, 1e-8)
    pt = max(math.sqrt(px * px + py * py), 1e-8)
    phi = math.atan2(py, px)
    eta = float(np.arcsinh(pz / max(pt, 1e-8)))
    eta = float(np.clip(eta, -5.0, 5.0))
    mass2 = max(E * E - (px * px + py * py + pz * pz), 0.0)
    # enforce E >= pt*cosh(eta)
    Emin = pt * math.cosh(eta)
    E = max(E, Emin, math.sqrt(max(mass2 + pt * pt * math.cosh(eta) ** 2, 0.0)))
    return np.array([pt, eta, phi, E], dtype=np.float32)


def _apply_corruption_batch(
    const_in: np.ndarray,
    mask_in: np.ndarray,
    kind: str,
    severity: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    const = const_in.copy()
    mask = mask_in.copy()
    B, L, _ = const.shape

    if kind == "pt_noise":
        z = rng.normal(0.0, float(severity), size=(B, L)).astype(np.float32)
        f = np.exp(z).astype(np.float32)
        m = mask
        const[:, :, 0] = np.where(m, const[:, :, 0] * f, const[:, :, 0])
        const[:, :, 3] = np.where(m, const[:, :, 3] * f, const[:, :, 3])

    elif kind == "eta_phi_jitter":
        de = rng.normal(0.0, float(severity), size=(B, L)).astype(np.float32)
        dp = rng.normal(0.0, float(severity), size=(B, L)).astype(np.float32)
        m = mask
        const[:, :, 1] = np.where(m, np.clip(const[:, :, 1] + de, -5.0, 5.0), const[:, :, 1])
        const[:, :, 2] = np.where(m, _wrap_phi(const[:, :, 2] + dp), const[:, :, 2])

    elif kind == "dropout":
        keep = mask.copy()
        drop = (rng.random(size=mask.shape) < float(severity)) & mask
        keep[drop] = False
        empty = ~keep.any(axis=1)
        if np.any(empty):
            for i in np.where(empty)[0]:
                idx = np.where(mask[i])[0]
                if idx.size > 0:
                    j = int(idx[np.argmax(const[i, idx, 0])])
                    keep[i, j] = True
        mask = keep
        const[~mask] = 0.0

    elif kind == "merge":
        # Random pair merges (lightweight approximation to additional merging).
        frac = float(np.clip(severity, 0.0, 0.95))
        for i in range(B):
            idx = np.where(mask[i])[0]
            n = int(idx.size)
            if n < 2:
                continue
            n_pairs = max(1, int(round(0.5 * frac * n)))
            n_pairs = min(n_pairs, n // 2)
            perm = rng.permutation(idx)
            for k in range(n_pairs):
                a = int(perm[2 * k])
                b = int(perm[2 * k + 1])
                merged = _merge_two_tokens(const[i, a], const[i, b])
                const[i, a] = merged
                const[i, b] = 0.0
                mask[i, b] = False

    elif kind == "global_scale":
        z = rng.normal(0.0, float(severity), size=(B, 1)).astype(np.float32)
        f = np.exp(z).astype(np.float32)
        m = mask
        const[:, :, 0] = np.where(m, const[:, :, 0] * f, const[:, :, 0])
        const[:, :, 3] = np.where(m, const[:, :, 3] * f, const[:, :, 3])

    else:
        raise ValueError(f"Unknown corruption kind: {kind}")

    # Physical consistency clamps.
    const[:, :, 0] = np.where(mask, np.clip(const[:, :, 0], 1e-8, 1e8), 0.0)
    const[:, :, 1] = np.where(mask, np.clip(const[:, :, 1], -5.0, 5.0), 0.0)
    const[:, :, 2] = np.where(mask, _wrap_phi(const[:, :, 2]), 0.0)
    e_floor = const[:, :, 0] * np.cosh(const[:, :, 1])
    const[:, :, 3] = np.where(mask, np.maximum(const[:, :, 3], e_floor), 0.0)
    return const.astype(np.float32), mask.astype(bool)


@dataclass
class InferOutput:
    p_hlt: np.ndarray
    p_joint: np.ndarray
    diag: Dict[str, np.ndarray]


def _infer_scores_and_diag(
    baseline: ParticleTransformer,
    reconstructor: OfflineReconstructor,
    dual: DualViewCrossAttnClassifier,
    feat_hlt_std: np.ndarray,
    mask_hlt: np.ndarray,
    const_hlt: np.ndarray,
    batch_size: int,
    device: torch.device,
    corrected_weight_floor: float,
    corrected_use_flags: bool,
    collect_diag: bool = True,
) -> InferOutput:
    n = int(feat_hlt_std.shape[0])
    p_h = np.zeros((n,), dtype=np.float32)
    p_j = np.zeros((n,), dtype=np.float32)

    diag: Dict[str, np.ndarray] = {}
    if collect_diag:
        diag = {
            "action_entropy_mean": np.zeros((n,), dtype=np.float32),
            "action_peak_mean": np.zeros((n,), dtype=np.float32),
            "action_highpeak_frac": np.zeros((n,), dtype=np.float32),
            "split_total_mean": np.zeros((n,), dtype=np.float32),
            "split_total_max": np.zeros((n,), dtype=np.float32),
            "split_child_sat_hi_frac": np.zeros((n,), dtype=np.float32),
            "budget_total": np.zeros((n,), dtype=np.float32),
            "budget_merge": np.zeros((n,), dtype=np.float32),
            "budget_eff": np.zeros((n,), dtype=np.float32),
            "budget_merge_pressure": np.zeros((n,), dtype=np.float32),
            "budget_eff_pressure": np.zeros((n,), dtype=np.float32),
            "gen_weight_mean": np.zeros((n,), dtype=np.float32),
        }

    baseline.eval()
    reconstructor.eval()
    dual.eval()

    ptr = 0
    with torch.no_grad():
        for s in range(0, n, int(batch_size)):
            e = min(n, s + int(batch_size))
            x_np = feat_hlt_std[s:e]
            m_np = mask_hlt[s:e]
            c_np = const_hlt[s:e]

            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(m_np).to(device=device, dtype=torch.bool)
            c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)

            logit_h = baseline(x, m).squeeze(1)
            out = reconstructor(x, m, c, stage_scale=1.0)
            feat_b, mask_b = joint_base.build_soft_corrected_view(
                out,
                weight_floor=float(corrected_weight_floor),
                scale_features_by_weight=True,
                include_flags=bool(corrected_use_flags),
            )
            logit_j = dual(x, m, feat_b, mask_b).squeeze(1)

            ph = torch.sigmoid(logit_h).detach().cpu().numpy().astype(np.float32)
            pj = torch.sigmoid(logit_j).detach().cpu().numpy().astype(np.float32)
            p_h[s:e] = ph
            p_j[s:e] = pj

            if collect_diag:
                tok_mask = m.float()
                token_den = tok_mask.sum(dim=1).clamp(min=1.0)

                ap = out["action_prob"].clamp(min=1e-8)
                ent = -(ap * torch.log(ap)).sum(dim=-1)
                peak = ap.max(dim=-1).values

                K = max(int(reconstructor.max_split_children), 1)
                B, L = m.shape
                child = out["child_weight"].view(B, L, K)
                split_total = child.sum(dim=-1)

                gen_w = out["gen_weight"]
                bm = out["budget_merge"]
                be = out["budget_eff"]
                btot = out["budget_total"]
                child_sum = out["child_weight"].sum(dim=1).clamp(min=1e-6)
                gen_sum = out["gen_weight"].sum(dim=1).clamp(min=1e-6)

                diag["action_entropy_mean"][s:e] = ((ent * tok_mask).sum(dim=1) / token_den).cpu().numpy().astype(np.float32)
                diag["action_peak_mean"][s:e] = ((peak * tok_mask).sum(dim=1) / token_den).cpu().numpy().astype(np.float32)
                diag["action_highpeak_frac"][s:e] = (((peak > 0.9).float() * tok_mask).sum(dim=1) / token_den).cpu().numpy().astype(np.float32)
                diag["split_total_mean"][s:e] = ((split_total * tok_mask).sum(dim=1) / token_den).cpu().numpy().astype(np.float32)
                diag["split_total_max"][s:e] = split_total.masked_fill(~m, 0.0).max(dim=1).values.cpu().numpy().astype(np.float32)
                diag["split_child_sat_hi_frac"][s:e] = (out["child_weight"] > 0.98).float().mean(dim=1).cpu().numpy().astype(np.float32)
                diag["budget_total"][s:e] = btot.cpu().numpy().astype(np.float32)
                diag["budget_merge"][s:e] = bm.cpu().numpy().astype(np.float32)
                diag["budget_eff"][s:e] = be.cpu().numpy().astype(np.float32)
                diag["budget_merge_pressure"][s:e] = (bm / child_sum).cpu().numpy().astype(np.float32)
                diag["budget_eff_pressure"][s:e] = (be / gen_sum).cpu().numpy().astype(np.float32)
                diag["gen_weight_mean"][s:e] = gen_w.mean(dim=1).cpu().numpy().astype(np.float32)

            ptr += (e - s)

    assert ptr == n
    return InferOutput(p_hlt=p_h, p_joint=p_j, diag=diag)


def _parse_corruptions(spec: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for tok in [x.strip() for x in str(spec).split(",") if x.strip()]:
        if ":" not in tok:
            raise ValueError(f"Invalid corruption token '{tok}', expected kind:severity")
        k, v = tok.split(":", 1)
        out.append((k.strip(), float(v)))
    if not out:
        raise ValueError("No valid corruptions parsed")
    return out


def _init_shift_acc(n: int) -> Dict[str, np.ndarray]:
    z = lambda: np.zeros((n,), dtype=np.float64)
    ninf = lambda: np.full((n,), -np.inf, dtype=np.float64)
    return {
        "count": z(),
        "abs_dp_sum_h": z(),
        "abs_dp_sumsq_h": z(),
        "abs_dp_max_h": ninf(),
        "abs_dp_sum_j": z(),
        "abs_dp_sumsq_j": z(),
        "abs_dp_max_j": ninf(),
        "dconf_sum_h": z(),
        "dconf_sumsq_h": z(),
        "dconf_max_h": ninf(),
        "dconf_sum_j": z(),
        "dconf_sumsq_j": z(),
        "dconf_max_j": ninf(),
        "dent_sum_h": z(),
        "dent_sumsq_h": z(),
        "dent_max_h": ninf(),
        "dent_sum_j": z(),
        "dent_sumsq_j": z(),
        "dent_max_j": ninf(),
        "dlogit_sum_h": z(),
        "dlogit_sumsq_h": z(),
        "dlogit_max_h": ninf(),
        "dlogit_sum_j": z(),
        "dlogit_sumsq_j": z(),
        "dlogit_max_j": ninf(),
        "js_sum_h": z(),
        "js_sumsq_h": z(),
        "js_max_h": ninf(),
        "js_sum_j": z(),
        "js_sumsq_j": z(),
        "js_max_j": ninf(),
        "flip_count_h": z(),
        "flip_count_j": z(),
        "joint_more_stable_prob_count": z(),
        "joint_more_stable_conf_count": z(),
        "disagree_growth_sum": z(),
        "disagree_growth_sumsq": z(),
        "disagree_growth_max": ninf(),
    }


def _update_shift_acc(
    acc: Dict[str, np.ndarray],
    p_h_clean: np.ndarray,
    p_j_clean: np.ndarray,
    p_h_corr: np.ndarray,
    p_j_corr: np.ndarray,
) -> None:
    ph0 = _clip_probs(p_h_clean)
    pj0 = _clip_probs(p_j_clean)
    ph1 = _clip_probs(p_h_corr)
    pj1 = _clip_probs(p_j_corr)

    dp_h = ph1 - ph0
    dp_j = pj1 - pj0
    adp_h = np.abs(dp_h)
    adp_j = np.abs(dp_j)

    conf0_h = np.abs(ph0 - 0.5) * 2.0
    conf1_h = np.abs(ph1 - 0.5) * 2.0
    conf0_j = np.abs(pj0 - 0.5) * 2.0
    conf1_j = np.abs(pj1 - 0.5) * 2.0
    dconf_h = conf0_h - conf1_h
    dconf_j = conf0_j - conf1_j

    ent0_h = _entropy_bernoulli(ph0)
    ent1_h = _entropy_bernoulli(ph1)
    ent0_j = _entropy_bernoulli(pj0)
    ent1_j = _entropy_bernoulli(pj1)
    dent_h = ent1_h - ent0_h
    dent_j = ent1_j - ent0_j

    dlogit_h = np.abs(_logit(ph1) - _logit(ph0))
    dlogit_j = np.abs(_logit(pj1) - _logit(pj0))

    js_h = _js_bernoulli(ph0, ph1)
    js_j = _js_bernoulli(pj0, pj1)

    flip_h = ((ph0 >= 0.5) != (ph1 >= 0.5)).astype(np.float64)
    flip_j = ((pj0 >= 0.5) != (pj1 >= 0.5)).astype(np.float64)

    disagree0 = np.abs(pj0 - ph0)
    disagree1 = np.abs(pj1 - ph1)
    ddis = disagree1 - disagree0

    acc["count"] += 1.0

    for pref, x in [("h", adp_h), ("j", adp_j)]:
        acc[f"abs_dp_sum_{pref}"] += x
        acc[f"abs_dp_sumsq_{pref}"] += x * x
        acc[f"abs_dp_max_{pref}"] = np.maximum(acc[f"abs_dp_max_{pref}"], x)

    for pref, x in [("h", dconf_h), ("j", dconf_j)]:
        acc[f"dconf_sum_{pref}"] += x
        acc[f"dconf_sumsq_{pref}"] += x * x
        acc[f"dconf_max_{pref}"] = np.maximum(acc[f"dconf_max_{pref}"], x)

    for pref, x in [("h", dent_h), ("j", dent_j)]:
        acc[f"dent_sum_{pref}"] += x
        acc[f"dent_sumsq_{pref}"] += x * x
        acc[f"dent_max_{pref}"] = np.maximum(acc[f"dent_max_{pref}"], np.abs(x))

    for pref, x in [("h", dlogit_h), ("j", dlogit_j)]:
        acc[f"dlogit_sum_{pref}"] += x
        acc[f"dlogit_sumsq_{pref}"] += x * x
        acc[f"dlogit_max_{pref}"] = np.maximum(acc[f"dlogit_max_{pref}"], x)

    for pref, x in [("h", js_h), ("j", js_j)]:
        acc[f"js_sum_{pref}"] += x
        acc[f"js_sumsq_{pref}"] += x * x
        acc[f"js_max_{pref}"] = np.maximum(acc[f"js_max_{pref}"], x)

    acc["flip_count_h"] += flip_h
    acc["flip_count_j"] += flip_j
    acc["joint_more_stable_prob_count"] += (adp_j < adp_h).astype(np.float64)
    acc["joint_more_stable_conf_count"] += (np.abs(dconf_j) < np.abs(dconf_h)).astype(np.float64)
    acc["disagree_growth_sum"] += ddis
    acc["disagree_growth_sumsq"] += ddis * ddis
    acc["disagree_growth_max"] = np.maximum(acc["disagree_growth_max"], np.abs(ddis))


def _std_from_sums(sum_v: np.ndarray, sumsq_v: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    mu = sum_v / np.maximum(cnt, 1.0)
    ex2 = sumsq_v / np.maximum(cnt, 1.0)
    var = np.maximum(ex2 - mu * mu, 0.0)
    return np.sqrt(var)


def _shift_features_from_acc(acc: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cnt = np.maximum(acc["count"], 1.0)
    f: Dict[str, np.ndarray] = {}

    for pref, pfx in [("h", "hlt"), ("j", "joint")]:
        f[f"{pfx}_mean_abs_dprob"] = (acc[f"abs_dp_sum_{pref}"] / cnt).astype(np.float32)
        f[f"{pfx}_std_abs_dprob"] = _std_from_sums(acc[f"abs_dp_sum_{pref}"], acc[f"abs_dp_sumsq_{pref}"], cnt).astype(np.float32)
        f[f"{pfx}_max_abs_dprob"] = np.maximum(acc[f"abs_dp_max_{pref}"], 0.0).astype(np.float32)

        f[f"{pfx}_mean_conf_drop"] = (acc[f"dconf_sum_{pref}"] / cnt).astype(np.float32)
        f[f"{pfx}_std_conf_drop"] = _std_from_sums(acc[f"dconf_sum_{pref}"], acc[f"dconf_sumsq_{pref}"], cnt).astype(np.float32)
        f[f"{pfx}_max_conf_drop"] = np.maximum(acc[f"dconf_max_{pref}"], 0.0).astype(np.float32)

        f[f"{pfx}_mean_delta_entropy"] = (acc[f"dent_sum_{pref}"] / cnt).astype(np.float32)
        f[f"{pfx}_std_delta_entropy"] = _std_from_sums(acc[f"dent_sum_{pref}"], acc[f"dent_sumsq_{pref}"], cnt).astype(np.float32)
        f[f"{pfx}_max_abs_delta_entropy"] = np.maximum(acc[f"dent_max_{pref}"], 0.0).astype(np.float32)

        f[f"{pfx}_mean_abs_dlogit"] = (acc[f"dlogit_sum_{pref}"] / cnt).astype(np.float32)
        f[f"{pfx}_std_abs_dlogit"] = _std_from_sums(acc[f"dlogit_sum_{pref}"], acc[f"dlogit_sumsq_{pref}"], cnt).astype(np.float32)
        f[f"{pfx}_max_abs_dlogit"] = np.maximum(acc[f"dlogit_max_{pref}"], 0.0).astype(np.float32)

        f[f"{pfx}_mean_js"] = (acc[f"js_sum_{pref}"] / cnt).astype(np.float32)
        f[f"{pfx}_std_js"] = _std_from_sums(acc[f"js_sum_{pref}"], acc[f"js_sumsq_{pref}"], cnt).astype(np.float32)
        f[f"{pfx}_max_js"] = np.maximum(acc[f"js_max_{pref}"], 0.0).astype(np.float32)

        f[f"{pfx}_flip_rate"] = (acc[f"flip_count_{pref}"] / cnt).astype(np.float32)

    f["joint_minus_hlt_mean_abs_dprob"] = (f["joint_mean_abs_dprob"] - f["hlt_mean_abs_dprob"]).astype(np.float32)
    f["joint_minus_hlt_mean_conf_drop"] = (f["joint_mean_conf_drop"] - f["hlt_mean_conf_drop"]).astype(np.float32)
    f["joint_minus_hlt_mean_delta_entropy"] = (f["joint_mean_delta_entropy"] - f["hlt_mean_delta_entropy"]).astype(np.float32)
    f["joint_minus_hlt_mean_js"] = (f["joint_mean_js"] - f["hlt_mean_js"]).astype(np.float32)
    f["joint_stability_prob_win_frac"] = (acc["joint_more_stable_prob_count"] / cnt).astype(np.float32)
    f["joint_stability_conf_win_frac"] = (acc["joint_more_stable_conf_count"] / cnt).astype(np.float32)
    f["mean_disagreement_growth"] = (acc["disagree_growth_sum"] / cnt).astype(np.float32)
    f["std_disagreement_growth"] = _std_from_sums(acc["disagree_growth_sum"], acc["disagree_growth_sumsq"], cnt).astype(np.float32)
    f["max_abs_disagreement_growth"] = np.maximum(acc["disagree_growth_max"], 0.0).astype(np.float32)
    return f


def _build_raw_features(
    p_h: np.ndarray,
    p_j: np.ndarray,
    diag: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    ph = _clip_probs(p_h)
    pj = _clip_probs(p_j)
    fh = np.abs(ph - 0.5) * 2.0
    fj = np.abs(pj - 0.5) * 2.0
    eh = _entropy_bernoulli(ph)
    ej = _entropy_bernoulli(pj)
    out: Dict[str, np.ndarray] = {
        "hlt_prob": ph.astype(np.float32),
        "joint_prob": pj.astype(np.float32),
        "hlt_logit": _logit(ph).astype(np.float32),
        "joint_logit": _logit(pj).astype(np.float32),
        "signed_disagree": (pj - ph).astype(np.float32),
        "abs_disagree": np.abs(pj - ph).astype(np.float32),
        "hlt_conf": fh.astype(np.float32),
        "joint_conf": fj.astype(np.float32),
        "conf_gap_j_minus_h": (fj - fh).astype(np.float32),
        "hlt_entropy": eh.astype(np.float32),
        "joint_entropy": ej.astype(np.float32),
        "entropy_gap_j_minus_h": (ej - eh).astype(np.float32),
        "agree_pred": ((ph >= 0.5) == (pj >= 0.5)).astype(np.float32),
    }
    for k, v in diag.items():
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def _stack_features(feature_dict: Dict[str, np.ndarray], feature_names: Sequence[str]) -> np.ndarray:
    cols = [np.asarray(feature_dict[k], dtype=np.float32).reshape(-1, 1) for k in feature_names]
    return np.concatenate(cols, axis=1)


def _lowfpr_cost(
    y: np.ndarray,
    p: np.ndarray,
    thr30: float,
    thr50: float,
    alpha_neg: float,
    tau: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    p = _clip_probs(p)
    bce = _bce_per_jet(y, p)
    neg = (1.0 - y)
    tail = _sigmoid((p - float(thr50)) / float(tau)) + 0.5 * _sigmoid((p - float(thr30)) / float(tau))
    return bce + float(alpha_neg) * neg * tail


def _score_metrics(y: np.ndarray, s: np.ndarray, hlt_ref: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(np.int64)
    s = _clip_probs(s)
    h = _clip_probs(hlt_ref)
    out = {
        "auc": float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float("nan"),
        "fpr30": _fpr_at_target_tpr(y, s, 0.30),
        "fpr50": _fpr_at_target_tpr(y, s, 0.50),
    }
    db = _bce_per_jet(y, s) - _bce_per_jet(y, h)
    out["harm_bce_frac_all_vs_hlt"] = float(np.mean(db > 0.0))
    neg = (y == 0)
    out["harm_bce_frac_neg_vs_hlt"] = float(np.mean(db[neg] > 0.0)) if np.any(neg) else float("nan")
    return out


def _choose_threshold_by_cal(
    y_cal: np.ndarray,
    p_h_cal: np.ndarray,
    p_j_cal: np.ndarray,
    gate_score_cal: np.ndarray,
    direction: str,
    objective: str = "fpr50",
) -> Tuple[float, Dict[str, float]]:
    vals = np.asarray(gate_score_cal, dtype=np.float64)
    qs = np.quantile(vals, np.linspace(0.02, 0.98, 97))
    best_t = float(qs[0])
    best = {"auc": -1.0, "fpr30": float("inf"), "fpr50": float("inf")}

    for t in qs:
        if direction == ">=":
            m = vals >= t
        else:
            m = vals <= t
        s = np.where(m, p_j_cal, p_h_cal)
        mtr = _score_metrics(y_cal, s, p_h_cal)
        if objective == "auc":
            key = (mtr["auc"], -mtr["fpr50"], -mtr["fpr30"])
            best_key = (best["auc"], -best["fpr50"], -best["fpr30"])
            if key > best_key:
                best_t = float(t)
                best = mtr
        else:
            key = (-mtr["fpr50"], mtr["auc"], -mtr["fpr30"])
            best_key = (-best["fpr50"], best["auc"], -best["fpr30"])
            if key > best_key:
                best_t = float(t)
                best = mtr
    return best_t, best


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Large-scale router signal sweep for HLT vs Joint routing (low-FPR focused).")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")
    ap.add_argument("--router_offset_jets", type=int, default=375000)
    ap.add_argument("--router_n_analysis", type=int, default=200000)
    ap.add_argument("--router_n_test", type=int, default=200000)
    ap.add_argument("--ref_offset_jets", type=int, default=0)
    ap.add_argument("--ref_n_jets", type=int, default=375000)
    ap.add_argument("--max_constits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hlt_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)
    ap.add_argument("--corruptions", type=str, default="pt_noise:0.03,pt_noise:0.06,eta_phi_jitter:0.02,eta_phi_jitter:0.05,dropout:0.05,dropout:0.10,merge:0.10,merge:0.20,global_scale:0.03")
    ap.add_argument("--router_cal_frac", type=float, default=0.2)
    ap.add_argument("--cost_alpha_neg", type=float, default=4.0)
    ap.add_argument("--cost_tau", type=float, default=0.02)
    ap.add_argument("--top_pair_signal_k", type=int, default=6)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    ap.add_argument("--save_per_jet_npz", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (run_dir / "router_signal_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(int(args.seed))

    data_setup_path = run_dir / "data_setup.json"
    if not data_setup_path.exists():
        raise FileNotFoundError(f"Missing data_setup.json in run_dir: {data_setup_path}")
    data_setup = json.loads(data_setup_path.read_text())

    cfg = _deepcopy_cfg()
    hlt_stats_path = run_dir / "hlt_stats.json"
    if hlt_stats_path.exists():
        hlt_obj = json.loads(hlt_stats_path.read_text())
        hcfg = hlt_obj.get("config", {})
        for k in list(cfg.get("hlt_effects", {}).keys()):
            if k in hcfg:
                cfg["hlt_effects"][k] = hcfg[k]

    train_files = _build_train_file_list(data_setup, args.train_path)

    # ---------------- Standardization stats from reference training setup ----------------
    ref_need = int(args.ref_offset_jets + args.ref_n_jets)
    const_ref_raw, labels_ref_all = load_raw_constituents_from_h5(
        train_files,
        max_jets=ref_need,
        max_constits=int(args.max_constits),
    )
    const_ref_raw = const_ref_raw[int(args.ref_offset_jets):int(args.ref_offset_jets) + int(args.ref_n_jets)]
    labels_ref = labels_ref_all[int(args.ref_offset_jets):int(args.ref_offset_jets) + int(args.ref_n_jets)].astype(np.int64)

    const_ref_off, mask_ref_off = _offline_mask(const_ref_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    feat_ref_off = compute_features(const_ref_off, mask_ref_off)
    idx_ref = np.arange(labels_ref.shape[0])
    tr_ref, _ = train_test_split(
        idx_ref,
        test_size=0.30,
        random_state=int(args.seed),
        stratify=labels_ref,
    )
    means, stds = get_stats(feat_ref_off, mask_ref_off, tr_ref)

    del const_ref_raw, labels_ref_all, labels_ref, const_ref_off, mask_ref_off, feat_ref_off, idx_ref, tr_ref
    gc.collect()

    # ---------------- Router dataset ----------------
    n_total_router = int(args.router_n_analysis + args.router_n_test)
    need = int(args.router_offset_jets + n_total_router)
    const_raw_all, labels_all = load_raw_constituents_from_h5(
        train_files,
        max_jets=need,
        max_constits=int(args.max_constits),
    )
    const_raw = const_raw_all[int(args.router_offset_jets):int(args.router_offset_jets) + n_total_router]
    labels = labels_all[int(args.router_offset_jets):int(args.router_offset_jets) + n_total_router].astype(np.int64)

    const_off, mask_off = _offline_mask(const_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    hlt_const, hlt_mask, hlt_stats_gen, _ = apply_hlt_effects_realistic_nomap(
        const_off,
        mask_off,
        cfg,
        seed=int(args.hlt_seed),
    )
    feat_hlt = compute_features(hlt_const, hlt_mask)
    feat_hlt_std = standardize(feat_hlt, hlt_mask, means, stds).astype(np.float32)

    # ---------------- Load models ----------------
    baseline_ckpt = run_dir / "baseline.pt"
    reco_ckpt = run_dir / "offline_reconstructor.pt"
    dual_ckpt = run_dir / "dual_joint.pt"
    if not baseline_ckpt.exists() or not reco_ckpt.exists() or not dual_ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoints in run_dir. Needed: baseline.pt, offline_reconstructor.pt, dual_joint.pt"
        )

    baseline_sd = _load_ckpt_state(baseline_ckpt, device)
    reco_sd = _load_ckpt_state(reco_ckpt, device)
    dual_sd = _load_ckpt_state(dual_ckpt, device)
    dual_in_a, dual_in_b = _infer_dual_input_dims(dual_sd)

    if int(dual_in_a) != 7:
        raise RuntimeError(
            f"This analysis currently supports dual input_dim_a=7 only. Found {dual_in_a}."
        )

    corrected_use_flags = bool(int(dual_in_b) == 12)

    baseline = ParticleTransformer(input_dim=7, **cfg["model"]).to(device)
    baseline.load_state_dict(baseline_sd, strict=True)

    reconstructor = OfflineReconstructor(input_dim=7, **cfg["reconstructor_model"]).to(device)
    reconstructor.load_state_dict(reco_sd, strict=True)

    dual = DualViewCrossAttnClassifier(input_dim_a=int(dual_in_a), input_dim_b=int(dual_in_b), **cfg["model"]).to(device)
    dual.load_state_dict(dual_sd, strict=True)

    # ---------------- Clean inference ----------------
    clean = _infer_scores_and_diag(
        baseline=baseline,
        reconstructor=reconstructor,
        dual=dual,
        feat_hlt_std=feat_hlt_std,
        mask_hlt=hlt_mask,
        const_hlt=hlt_const,
        batch_size=int(args.batch_size),
        device=device,
        corrected_weight_floor=float(args.corrected_weight_floor),
        corrected_use_flags=corrected_use_flags,
        collect_diag=True,
    )

    # ---------------- Corruption sweep ----------------
    corruption_list = _parse_corruptions(args.corruptions)
    acc = _init_shift_acc(n_total_router)

    for kind, severity in corruption_list:
        print(f"[corruption] kind={kind} severity={severity:.4f}")
        p_h_corr = np.zeros((n_total_router,), dtype=np.float32)
        p_j_corr = np.zeros((n_total_router,), dtype=np.float32)

        for s in range(0, n_total_router, int(args.batch_size)):
            e = min(n_total_router, s + int(args.batch_size))
            c_b, m_b = _apply_corruption_batch(hlt_const[s:e], hlt_mask[s:e], kind, severity, rng)
            f_b = compute_features(c_b, m_b)
            f_b_std = standardize(f_b, m_b, means, stds).astype(np.float32)

            out_b = _infer_scores_and_diag(
                baseline=baseline,
                reconstructor=reconstructor,
                dual=dual,
                feat_hlt_std=f_b_std,
                mask_hlt=m_b,
                const_hlt=c_b,
                batch_size=max(32, int(args.batch_size)),
                device=device,
                corrected_weight_floor=float(args.corrected_weight_floor),
                corrected_use_flags=corrected_use_flags,
                collect_diag=False,
            )
            p_h_corr[s:e] = out_b.p_hlt
            p_j_corr[s:e] = out_b.p_joint

        _update_shift_acc(acc, clean.p_hlt, clean.p_joint, p_h_corr, p_j_corr)

    shift_feats = _shift_features_from_acc(acc)

    # ---------------- Assemble feature table ----------------
    feature_dict = _build_raw_features(clean.p_hlt, clean.p_joint, clean.diag)
    feature_dict.update(shift_feats)
    feature_names = sorted(feature_dict.keys())
    X_all = _stack_features(feature_dict, feature_names)

    # ---------------- Split analysis/test ----------------
    idx_all = np.arange(n_total_router)
    idx_analysis, idx_test = train_test_split(
        idx_all,
        test_size=float(args.router_n_test) / float(n_total_router),
        random_state=int(args.seed),
        stratify=labels,
    )
    if len(idx_analysis) != int(args.router_n_analysis) or len(idx_test) != int(args.router_n_test):
        raise RuntimeError(
            f"Split mismatch: got analysis={len(idx_analysis)} test={len(idx_test)}; "
            f"expected {args.router_n_analysis}/{args.router_n_test}"
        )

    y_a = labels[idx_analysis]
    y_t = labels[idx_test]
    ph_a = clean.p_hlt[idx_analysis]
    pj_a = clean.p_joint[idx_analysis]
    ph_t = clean.p_hlt[idx_test]
    pj_t = clean.p_joint[idx_test]

    X_a = X_all[idx_analysis]
    X_t = X_all[idx_test]

    idx_fit, idx_cal = train_test_split(
        np.arange(len(idx_analysis)),
        test_size=float(args.router_cal_frac),
        random_state=int(args.seed) + 17,
        stratify=y_a,
    )

    y_fit = y_a[idx_fit]
    y_cal = y_a[idx_cal]
    X_fit = X_a[idx_fit]
    X_cal = X_a[idx_cal]
    ph_fit = ph_a[idx_fit]
    pj_fit = pj_a[idx_fit]
    ph_cal = ph_a[idx_cal]
    pj_cal = pj_a[idx_cal]

    # Low-FPR-weighted routing labels.
    thr_h30 = _threshold_at_target_tpr(y_fit, ph_fit, 0.30)
    thr_h50 = _threshold_at_target_tpr(y_fit, ph_fit, 0.50)
    thr_j30 = _threshold_at_target_tpr(y_fit, pj_fit, 0.30)
    thr_j50 = _threshold_at_target_tpr(y_fit, pj_fit, 0.50)

    c_h_fit = _lowfpr_cost(y_fit, ph_fit, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_fit = _lowfpr_cost(y_fit, pj_fit, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_fit = (c_j_fit < c_h_fit).astype(np.int64)

    c_h_cal = _lowfpr_cost(y_cal, ph_cal, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_cal = _lowfpr_cost(y_cal, pj_cal, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_cal = (c_j_cal < c_h_cal).astype(np.int64)

    c_h_test = _lowfpr_cost(y_t, ph_t, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_test = _lowfpr_cost(y_t, pj_t, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_test = (c_j_test < c_h_test).astype(np.int64)

    # ---------------- Baselines ----------------
    leaderboard: List[Dict[str, object]] = []

    def add_method(name: str, score_test: np.ndarray, extra: Dict[str, object] | None = None) -> None:
        row: Dict[str, object] = {"method": name}
        row.update(_score_metrics(y_t, score_test, ph_t))
        if extra:
            row.update(extra)
        leaderboard.append(row)

    add_method("hlt", ph_t)
    add_method("joint", pj_t)
    add_method("oracle_best_of_two_lowfpr_cost", np.where(z_test == 1, pj_t, ph_t), {"oracle": 1})

    # Legacy raw-feature gate (as reference to prior marginal analysis)
    raw_gate_names = [
        "hlt_prob", "joint_prob", "signed_disagree", "abs_disagree", "hlt_conf", "joint_conf", "conf_gap_j_minus_h", "agree_pred"
    ]
    X_fit_raw = _stack_features({k: feature_dict[k][idx_analysis][idx_fit] for k in raw_gate_names}, raw_gate_names)
    X_cal_raw = _stack_features({k: feature_dict[k][idx_analysis][idx_cal] for k in raw_gate_names}, raw_gate_names)
    X_test_raw = _stack_features({k: feature_dict[k][idx_test] for k in raw_gate_names}, raw_gate_names)

    lg_raw = LogisticRegression(C=0.2, max_iter=3000, class_weight="balanced")
    lg_raw.fit(X_fit_raw, z_fit)
    q_cal_raw = lg_raw.predict_proba(X_cal_raw)[:, 1]
    q_test_raw = lg_raw.predict_proba(X_test_raw)[:, 1]
    add_method("legacy_softmix_raw", q_test_raw * pj_t + (1.0 - q_test_raw) * ph_t)
    add_method("legacy_hard_raw", np.where(q_test_raw >= 0.5, pj_t, ph_t), {"joint_route_frac": float(np.mean(q_test_raw >= 0.5))})

    # ---------------- Single-threshold hard rules ----------------
    best_single = None
    for fn in feature_names:
        s_fit = feature_dict[fn][idx_analysis][idx_fit]
        s_cal = feature_dict[fn][idx_analysis][idx_cal]
        s_test = feature_dict[fn][idx_test]
        for direction in [">=", "<="]:
            t, _ = _choose_threshold_by_cal(y_cal, ph_cal, pj_cal, s_cal, direction, objective="fpr50")
            m_test = (s_test >= t) if direction == ">=" else (s_test <= t)
            p_test = np.where(m_test, pj_t, ph_t)
            mtr = _score_metrics(y_t, p_test, ph_t)
            cand = {
                "method": "hard_single_rule",
                "feature": fn,
                "direction": direction,
                "threshold": float(t),
                "joint_route_frac": float(np.mean(m_test)),
                **mtr,
            }
            if best_single is None:
                best_single = cand
            else:
                key = (-cand["fpr50"], cand["auc"], -cand["fpr30"])
                best_key = (-best_single["fpr50"], best_single["auc"], -best_single["fpr30"])
                if key > best_key:
                    best_single = cand
    if best_single is not None:
        leaderboard.append(best_single)

    # ---------------- Pairwise hard rules on top-K univariate signals ----------------
    # rank signals by fit winner AUC
    sig_rank: List[Tuple[float, str]] = []
    for fn in feature_names:
        s = feature_dict[fn][idx_analysis][idx_fit]
        if np.allclose(np.std(s), 0.0):
            continue
        try:
            auc = roc_auc_score(z_fit, s)
            auc = max(float(auc), 1.0 - float(auc))
        except Exception:
            auc = 0.5
        sig_rank.append((auc, fn))
    sig_rank.sort(reverse=True)
    top_names = [x[1] for x in sig_rank[: max(2, int(args.top_pair_signal_k))]]

    best_pair = None
    for i, f1 in enumerate(top_names):
        for f2 in top_names[i + 1 :]:
            a_cal = feature_dict[f1][idx_analysis][idx_cal]
            b_cal = feature_dict[f2][idx_analysis][idx_cal]
            a_test = feature_dict[f1][idx_test]
            b_test = feature_dict[f2][idx_test]
            qa = np.quantile(feature_dict[f1][idx_analysis][idx_fit], np.linspace(0.1, 0.9, 9))
            qb = np.quantile(feature_dict[f2][idx_analysis][idx_fit], np.linspace(0.1, 0.9, 9))
            for da in [">=", "<="]:
                for db in [">=", "<="]:
                    for ta in qa:
                        ma_cal = a_cal >= ta if da == ">=" else a_cal <= ta
                        ma_test = a_test >= ta if da == ">=" else a_test <= ta
                        for tb in qb:
                            mb_cal = b_cal >= tb if db == ">=" else b_cal <= tb
                            mb_test = b_test >= tb if db == ">=" else b_test <= tb
                            m_cal = ma_cal & mb_cal
                            p_cal = np.where(m_cal, pj_cal, ph_cal)
                            cal = _score_metrics(y_cal, p_cal, ph_cal)

                            m_test = ma_test & mb_test
                            p_test = np.where(m_test, pj_t, ph_t)
                            mtr = _score_metrics(y_t, p_test, ph_t)
                            cand = {
                                "method": "hard_pair_rule",
                                "f1": f1,
                                "d1": da,
                                "t1": float(ta),
                                "f2": f2,
                                "d2": db,
                                "t2": float(tb),
                                "joint_route_frac": float(np.mean(m_test)),
                                "cal_fpr50": float(cal["fpr50"]),
                                **mtr,
                            }
                            if best_pair is None:
                                best_pair = cand
                            else:
                                key = (-cand["fpr50"], cand["auc"], -cand["fpr30"])
                                best_key = (-best_pair["fpr50"], best_pair["auc"], -best_pair["fpr30"])
                                if key > best_key:
                                    best_pair = cand
    if best_pair is not None:
        leaderboard.append(best_pair)

    # ---------------- Learned routers ----------------
    models = [
        ("logreg", LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)),
        ("tree_d2", DecisionTreeClassifier(max_depth=2, min_samples_leaf=400, class_weight="balanced", random_state=args.seed)),
        ("tree_d3", DecisionTreeClassifier(max_depth=3, min_samples_leaf=400, class_weight="balanced", random_state=args.seed)),
        ("mlp_64", MLPClassifier(hidden_layer_sizes=(64,), max_iter=250, alpha=1e-4, random_state=args.seed)),
    ]

    for name, clf in models:
        clf.fit(X_fit, z_fit)
        if hasattr(clf, "predict_proba"):
            q_cal = clf.predict_proba(X_cal)[:, 1]
            q_test = clf.predict_proba(X_t)[:, 1]
        else:
            zc = clf.decision_function(X_cal)
            zt = clf.decision_function(X_t)
            q_cal = _sigmoid(zc)
            q_test = _sigmoid(zt)

        # hard route with tuned threshold
        t_hard, _ = _choose_threshold_by_cal(y_cal, ph_cal, pj_cal, q_cal, ">=", objective="fpr50")
        hard_mask = q_test >= t_hard
        add_method(
            f"{name}_hard",
            np.where(hard_mask, pj_t, ph_t),
            {"joint_route_frac": float(np.mean(hard_mask)), "q_threshold": float(t_hard)},
        )

        # softmix route
        add_method(f"{name}_softmix", q_test * pj_t + (1.0 - q_test) * ph_t)

    # ---------------- No-harm gate (HLT default) ----------------
    # Reuse a logreg gate; route to joint only with strong evidence and only when HLT is uncertain.
    nh = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)
    nh.fit(X_fit, z_fit)
    q_cal = nh.predict_proba(X_cal)[:, 1]
    q_test = nh.predict_proba(X_t)[:, 1]
    conf_h_cal = np.abs(ph_cal - 0.5) * 2.0
    conf_h_test = np.abs(ph_t - 0.5) * 2.0

    best_nh = None
    for t in np.linspace(0.55, 0.98, 30):
        for cmax in np.linspace(0.20, 0.90, 15):
            m_cal = (q_cal >= t) & (conf_h_cal <= cmax)
            s_cal = np.where(m_cal, pj_cal, ph_cal)
            cal = _score_metrics(y_cal, s_cal, ph_cal)
            # no-harm-ish constraint on cal
            if cal["fpr50"] > _fpr_at_target_tpr(y_cal, ph_cal, 0.50) + 0.0005:
                continue
            m_test = (q_test >= t) & (conf_h_test <= cmax)
            s_test = np.where(m_test, pj_t, ph_t)
            mtr = _score_metrics(y_t, s_test, ph_t)
            cand = {
                "method": "no_harm_gate",
                "q_threshold": float(t),
                "hlt_conf_max": float(cmax),
                "joint_route_frac": float(np.mean(m_test)),
                "cal_fpr50": float(cal["fpr50"]),
                **mtr,
            }
            if best_nh is None:
                best_nh = cand
            else:
                key = (-cand["fpr50"], cand["auc"], -cand["fpr30"])
                best_key = (-best_nh["fpr50"], best_nh["auc"], -best_nh["fpr30"])
                if key > best_key:
                    best_nh = cand

    if best_nh is not None:
        leaderboard.append(best_nh)

    # ---------------- Signal ranking (fit split) ----------------
    signal_rank_rows: List[Dict[str, object]] = []
    delta_cost_fit = c_j_fit - c_h_fit
    for fn in feature_names:
        s = feature_dict[fn][idx_analysis][idx_fit]
        if np.allclose(np.std(s), 0.0):
            auc_win = 0.5
            corr = 0.0
        else:
            try:
                auc_win = roc_auc_score(z_fit, s)
                auc_win = max(float(auc_win), 1.0 - float(auc_win))
            except Exception:
                auc_win = 0.5
            c = np.corrcoef(s, delta_cost_fit)[0, 1]
            corr = 0.0 if not np.isfinite(c) else float(c)
        signal_rank_rows.append({
            "feature": fn,
            "winner_auc_abs": float(auc_win),
            "pearson_with_delta_cost": float(corr),
        })
    signal_rank_rows.sort(key=lambda r: (r["winner_auc_abs"], abs(r["pearson_with_delta_cost"])), reverse=True)

    # ---------------- Final report ----------------
    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda r: (float("inf") if (not np.isfinite(float(r.get("fpr50", np.nan)))) else float(r["fpr50"]), -float(r.get("auc", -1.0))),
    )

    report = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "settings": {
            "router_offset_jets": int(args.router_offset_jets),
            "router_n_analysis": int(args.router_n_analysis),
            "router_n_test": int(args.router_n_test),
            "ref_n_jets": int(args.ref_n_jets),
            "max_constits": int(args.max_constits),
            "seed": int(args.seed),
            "hlt_seed": int(args.hlt_seed),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "corrected_weight_floor": float(args.corrected_weight_floor),
            "corruptions": [{"kind": k, "severity": float(v)} for (k, v) in corruption_list],
            "cost_alpha_neg": float(args.cost_alpha_neg),
            "cost_tau": float(args.cost_tau),
            "dual_input_dim_a": int(dual_in_a),
            "dual_input_dim_b": int(dual_in_b),
            "corrected_use_flags": bool(corrected_use_flags),
            "n_total_router": int(n_total_router),
            "n_analysis": int(len(idx_analysis)),
            "n_test": int(len(idx_test)),
            "n_fit": int(len(idx_fit)),
            "n_cal": int(len(idx_cal)),
        },
        "clean_metrics": {
            "hlt_test": _score_metrics(y_t, ph_t, ph_t),
            "joint_test": _score_metrics(y_t, pj_t, ph_t),
            "oracle_best_of_two_lowfpr_cost_test": _score_metrics(y_t, np.where(z_test == 1, pj_t, ph_t), ph_t),
        },
        "leaderboard": leaderboard_sorted,
        "top_signal_rank": signal_rank_rows[:80],
        "artifacts": {
            "leaderboard_csv": str((out_dir / "router_leaderboard_test.csv").resolve()),
            "signal_rank_csv": str((out_dir / "router_signal_rank_fit.csv").resolve()),
        },
    }

    rep_path = Path(args.report_json).expanduser().resolve() if str(args.report_json).strip() else (out_dir / "router_signal_sweep_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w") as f:
        json.dump(report, f, indent=2)

    _save_csv(out_dir / "router_leaderboard_test.csv", leaderboard_sorted)
    _save_csv(out_dir / "router_signal_rank_fit.csv", signal_rank_rows)

    if bool(args.save_per_jet_npz):
        np.savez_compressed(
            out_dir / "router_signal_perjet.npz",
            labels=labels.astype(np.int8),
            idx_analysis=idx_analysis.astype(np.int64),
            idx_test=idx_test.astype(np.int64),
            preds_hlt=clean.p_hlt.astype(np.float32),
            preds_joint=clean.p_joint.astype(np.float32),
            **{k: np.asarray(v, dtype=np.float32) for k, v in feature_dict.items()},
        )

    print("=" * 72)
    print("Router Signal Sweep")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {out_dir}")
    print(f"N analysis/test: {len(idx_analysis)} / {len(idx_test)}")
    print(f"Corruptions: {len(corruption_list)}")
    print()
    print("Top methods on held-out test (sorted by FPR@50 then AUC):")
    for r in leaderboard_sorted[:12]:
        print(
            f"  {str(r.get('method')):28s} "
            f"AUC={float(r.get('auc', float('nan'))):.6f} "
            f"FPR30={float(r.get('fpr30', float('nan'))):.6f} "
            f"FPR50={float(r.get('fpr50', float('nan'))):.6f} "
            f"harm_neg={float(r.get('harm_bce_frac_neg_vs_hlt', float('nan'))):.4f}"
        )
    print()
    print(f"Saved report: {rep_path}")
    print(f"Saved leaderboard CSV: {out_dir / 'router_leaderboard_test.csv'}")
    print(f"Saved signal-rank CSV: {out_dir / 'router_signal_rank_fit.csv'}")
    if bool(args.save_per_jet_npz):
        print(f"Saved per-jet NPZ: {out_dir / 'router_signal_perjet.npz'}")


if __name__ == "__main__":
    main()
