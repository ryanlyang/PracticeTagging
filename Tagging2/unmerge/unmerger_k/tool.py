"""
Tools for Ordered + k_pred unmerger experiment.

Key idea:
- Avoid storing a full `groups[N][S]` python structure (too memory heavy).
- Instead, during the merge step we record ONLY merged-parent samples:
    samples = [(jet_idx, parent_idx), ...]
    children = [np.ndarray(child_indices), ...]
  This is sufficient for teacher-forced reconstruction training.

Targets:
- parent_gt[j,i] = 1 if token i survives HLT and has group_size>1
- k_true token-level can be:
    - "missing": group_size-1
    - "children": group_size
- reconstruction target per sample: pt-sorted child features packed to Kmax.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve


# -----------------------------
# Simple experiment I/O helpers
# -----------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(config: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return p


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, p.as_posix())
    return p


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    p = Path(path)
    payload = torch.load(p.as_posix(), map_location=map_location)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=bool(strict))
    return payload if isinstance(payload, dict) else {"state_dict": state}


# -----------------------------
# Geometry helpers + feature conversions
# -----------------------------


def wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def compute_jet_axis(const: np.ndarray, mask: np.ndarray, eps: float = 1e-8):
    """Compute jet axis from (pt, eta, phi, E) using sum of 4-vectors."""
    pt = np.maximum(const[:, :, 0], eps)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], eps)

    m = mask.astype(np.float32)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    jet_px = (px * m).sum(axis=1)
    jet_py = (py * m).sum(axis=1)
    jet_pz = (pz * m).sum(axis=1)
    jet_E = (E * m).sum(axis=1)

    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + eps
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + eps), eps, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + eps

    return (
        jet_eta.astype(np.float32),
        jet_phi.astype(np.float32),
        jet_pt.astype(np.float32),
        jet_E.astype(np.float32),
    )


def raw_to_feats_with_axis(const: np.ndarray, mask: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> np.ndarray:
    """Convert raw (pt,eta,phi,E) -> (log_pt, dEta, dPhi, log_E) using provided axis."""
    eps = 1e-8
    pt = np.maximum(const[:, :, 0], eps)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], eps)

    dEta = np.clip(eta - jet_eta[:, None], -5.0, 5.0)
    dPhi = wrap_dphi(phi - jet_phi[:, None])
    log_pt = np.log(pt)
    log_E = np.log(E)
    feats = np.stack([log_pt, dEta, dPhi, log_E], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats[~mask] = 0.0
    return feats.astype(np.float32)


def raw_to_feats7_with_axis(
    const: np.ndarray,
    mask: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    jet_pt: np.ndarray,
    jet_E: np.ndarray,
) -> np.ndarray:
    """Compute 7 engineered features in jet-axis frame (mirrors Baseline.compute_features)."""
    eps = 1e-8
    pt = np.maximum(const[:, :, 0], eps)
    eta = np.clip(const[:, :, 1], -5, 5)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], eps)

    dEta = np.clip(eta - jet_eta[:, None], -5.0, 5.0)
    dPhi = wrap_dphi(phi - jet_phi[:, None])
    log_pt = np.log(pt + eps)
    log_E = np.log(E + eps)
    log_pt_over_jetpt = np.log(pt / np.maximum(jet_pt[:, None], eps) + eps)
    log_E_over_jetE = np.log(E / np.maximum(jet_E[:, None], eps) + eps)
    dR = np.sqrt(dEta**2 + dPhi**2)
    feats = np.stack(
        [dEta, dPhi, log_pt, log_E, log_pt_over_jetpt, log_E_over_jetE, dR],
        axis=-1,
    )
    feats = np.clip(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    feats[~mask] = 0.0
    return feats.astype(np.float32)


def raw_to_feats(
    const: np.ndarray,
    mask: np.ndarray,
    hlt_axis: np.ndarray,
    *,
    kind: str = "4d",
) -> np.ndarray:
    """Feature front-end switch: kind in {"4d","7d"}."""
    jet_eta = hlt_axis[:, 0]
    jet_phi = hlt_axis[:, 1]
    jet_pt = hlt_axis[:, 2]
    jet_E = hlt_axis[:, 3]
    k = str(kind).lower()
    if k in ("4", "4d", "raw4", "logpt_detadphi_loge"):
        return raw_to_feats_with_axis(const, mask, jet_eta, jet_phi)
    if k in ("7", "7d", "eng7", "engineered7"):
        return raw_to_feats7_with_axis(const, mask, jet_eta, jet_phi, jet_pt, jet_E)
    raise ValueError(f"Unknown feature kind: {kind}")


def feats_to_raw(feats: np.ndarray, jet_eta: np.ndarray, jet_phi: np.ndarray) -> np.ndarray:
    """Map features back to raw (pt,eta,phi,E). Supports 4D and the Baseline-style 7D engineered features."""
    eps = 1e-8
    D = int(feats.shape[-1])
    if D == 4:
        # (log_pt, dEta, dPhi, log_E)
        log_pt = feats[..., 0]
        dEta = feats[..., 1]
        dPhi = feats[..., 2]
        log_E = feats[..., 3]
    elif D >= 7:
        # Baseline 7D order: (dEta, dPhi, log_pt, log_E, log(pt/jet_pt), log(E/jet_E), dR)
        dEta = feats[..., 0]
        dPhi = feats[..., 1]
        log_pt = feats[..., 2]
        log_E = feats[..., 3]
    else:
        raise ValueError(f"Unsupported feature dim for feats_to_raw: {D}")

    pt = np.maximum(np.exp(log_pt), eps)
    eta = np.clip(jet_eta[..., None] + dEta, -5, 5)
    phi = np.arctan2(np.sin(jet_phi[..., None] + dPhi), np.cos(jet_phi[..., None] + dPhi))
    E = np.maximum(np.exp(log_E), eps)
    raw = np.stack([pt, eta, phi, E], axis=-1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw.astype(np.float32)


def get_stats_tokens(feat: np.ndarray, mask: np.ndarray, idx_sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = feat.shape[-1]
    means = np.zeros(D, dtype=np.float64)
    stds = np.zeros(D, dtype=np.float64)
    for d in range(D):
        vals = feat[idx_sel, :, d][mask[idx_sel]]
        means[d] = float(np.nanmean(vals))
        stds[d] = float(np.nanstd(vals) + 1e-8)
    return means.astype(np.float32), stds.astype(np.float32)


def standardize_tokens(feat: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray, clip: float = 10.0) -> np.ndarray:
    std = (feat - means[None, None, :]) / stds[None, None, :]
    std = np.clip(std, -float(clip), float(clip))
    std = np.nan_to_num(std, 0.0)
    std[~mask] = 0.0
    return std.astype(np.float32)


# -----------------------------
# HLT simulation with compact merge samples
# -----------------------------


def apply_hlt_effects_collect_samples(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: dict,
    *,
    seed: int = 42,
    k_max: int = 8,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Tuple[int, int]],
    List[np.ndarray],
]:
    """
    Apply HLT effects (threshold+merge+smear+eff).

    Returns:
      hlt_raw: [N,S,4]
      hlt_mask: [N,S]
      off_mask: [N,S] (offline threshold on raw)
      group_size: [N,S] int16 (0 for masked tokens)
      hlt_axis: [N,4] float32 (jet_eta, jet_phi, jet_pt, jet_E)
      samples: list of (jet_idx, parent_idx) for merged parents (group_size>1)
      children: list of child-index arrays (pt-sorted, truncated to k_max) matching `samples`
    """
    np.random.seed(int(seed))
    hcfg = cfg["hlt_effects"]
    n_jets, S, _ = const.shape
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Offline threshold mask (reference)
    pt_thr_off = float(hcfg["pt_threshold_offline"])
    off_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)

    # HLT threshold
    pt_thr_hlt = float(hcfg["pt_threshold_hlt"])
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0

    group_size = np.zeros((n_jets, S), dtype=np.int16)
    samples: List[Tuple[int, int]] = []  #记录第几个jet的第几个contituent是parent（after efficiency loss）
    children: List[np.ndarray] = []  #对应samples的child contituents的索引

    # Merge (record only merged parents)  
    # group_size记录jet中每个contituent的merge数量
    # （0表示被merge，1表示未被merge但也没merge其它粒子，n表示merge了n-1个粒子加它自己就是n个粒子），
    #  g记录哪些contituents(索引)被merge到一起
    if bool(hcfg["merge_enabled"]) and float(hcfg["merge_radius"]) > 0:
        r = float(hcfg["merge_radius"])
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]  #返回每个jet中有效的contituents的索引
            if len(valid_idx) == 0:
                continue
            g: dict[int, List[int]] = {int(ii): [int(ii)] for ii in valid_idx}
            # 初始化group_size，每个有效的contituent的group_size为1
            for ii in valid_idx:
                group_size[jet_idx, int(ii)] = 1

            if len(valid_idx) >= 2:
                to_remove = set()  #记录需要被移除的contituents的索引
                for a in range(len(valid_idx)):
                    i = int(valid_idx[a])
                    if i in to_remove:
                        #如果i已经在to_remove中，则跳过
                        continue
                    for b in range(a + 1, len(valid_idx)):
                        j = int(valid_idx[b])
                        if j in to_remove:
                            #如果j已经在to_remove中，则跳过
                            continue
                        #计算i和j的dEta和dPhi
                        deta = float(hlt[jet_idx, i, 1] - hlt[jet_idx, j, 1])
                        dphi = float(wrap_dphi(hlt[jet_idx, i, 2] - hlt[jet_idx, j, 2]))
                        dR = math.sqrt(deta * deta + dphi * dphi)
                        if dR < r:
                            pt_i = float(hlt[jet_idx, i, 0])
                            pt_j = float(hlt[jet_idx, j, 0])
                            pt_sum = pt_i + pt_j
                            if pt_sum < 1e-6:
                                continue
                            g[i].extend(g[j]) #把g[j]的contituents的索引添加到g[i]中
                            g.pop(j, None) #删除key为j的这一列
                            group_size[jet_idx, i] = int(group_size[jet_idx, i] + group_size[jet_idx, j]) #更新group_size
                            group_size[jet_idx, j] = 0 #group_size[jet_idx, j]设置为0表示被merge

                            # pT-weighted merge in HLT raw space, 计算合并后的pt, eta, phi, E(E 在 smearing 阶段会被重新用 (pt,eta) 计算覆盖)
                            w_i, w_j = pt_i / pt_sum, pt_j / pt_sum
                            hlt[jet_idx, i, 0] = pt_sum
                            hlt[jet_idx, i, 1] = w_i * hlt[jet_idx, i, 1] + w_j * hlt[jet_idx, j, 1]
                            phi_i, phi_j = float(hlt[jet_idx, i, 2]), float(hlt[jet_idx, j, 2])
                            hlt[jet_idx, i, 2] = math.atan2(
                                w_i * math.sin(phi_i) + w_j * math.sin(phi_j),
                                w_i * math.cos(phi_i) + w_j * math.cos(phi_j),
                            )
                            hlt[jet_idx, i, 3] = float(hlt[jet_idx, i, 3]) + float(hlt[jet_idx, j, 3])
                            to_remove.add(j) #记录需要被移除的contituents的索引

                for idx in to_remove:
                    #将需要被移除的contituents的mask设置为False，即被merge
                    hlt_mask[jet_idx, idx] = False
                    #将需要被移除的contituents的raw设置为0，即被merge
                    hlt[jet_idx, idx] = 0

            # Record merged parents for this jet
            for p, ch in g.items():
                if not bool(hlt_mask[jet_idx, p]):
                    continue
                if len(ch) <= 1:
                    continue
                ch_arr = np.asarray(ch, dtype=np.int64)
                # Sort children by OFFLINE pt (raw const)
                pts = const[jet_idx, ch_arr, 0]
                order = np.argsort(-pts)
                ch_arr = ch_arr[order]
                if int(ch_arr.shape[0]) > int(k_max):
                    ch_arr = ch_arr[: int(k_max)]  #只保留前k_max个contituents
                samples.append((int(jet_idx), int(p)))  #记录第几个jet的第几个contituent是parent
                children.append(ch_arr)
    else:
        # No merge: group_size is 1 for valid tokens, but there are no merged-parent samples.
        for jet_idx in range(n_jets):
            valid_idx = np.where(hlt_mask[jet_idx])[0]
            group_size[jet_idx, valid_idx] = 1

    # Smearing
    valid = hlt_mask.copy()
    pt_noise = np.clip(
        np.random.normal(1.0, float(hcfg.get("pt_resolution", 0.0)), (n_jets, S)),
        0.5,
        1.5,
    )
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0)
    eta_noise = np.random.normal(0, float(hcfg.get("eta_resolution", 0.0)), (n_jets, S))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5, 5), 0)
    phi_noise = np.random.normal(0, float(hcfg.get("phi_resolution", 0.0)), (n_jets, S))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0)
    # Recompute E from (pt, eta)
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5, 5)), 0)

    # Efficiency loss
    eff = float(hcfg.get("efficiency_loss", 0.0))
    if eff > 0:
        lost = (np.random.random((n_jets, S)) < eff) & hlt_mask
        hlt_mask[lost] = False
        hlt[lost] = 0
        group_size[lost] = 0

    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    jet_eta, jet_phi, jet_pt, jet_E = compute_jet_axis(hlt, hlt_mask, eps=1e-8)
    hlt_axis = np.stack([jet_eta, jet_phi, jet_pt, jet_E], axis=1).astype(np.float32)

    # Filter any samples whose parent token was later removed (e.g., by efficiency loss).
    if len(samples) > 0:
        s2: List[Tuple[int, int]] = []
        c2: List[np.ndarray] = []
        for (j, p), ch in zip(samples, children):
            if bool(hlt_mask[int(j), int(p)]):
                s2.append((int(j), int(p))) #如果parent token没有被efficiency loss移除，则记录第几个jet的第几个contituent是parent
                c2.append(ch) #同上，记录对应的child contituents的索引
        samples, children = s2, c2

    return hlt, hlt_mask, off_mask, group_size, hlt_axis, samples, children


# -----------------------------
# Targets + datasets
# -----------------------------


def build_parent_targets_from_group_size(
    hlt_mask: np.ndarray,
    group_size: np.ndarray,
    *,
    count_kind: str = "missing",  # "missing" => size-1, "children" => size
    max_k: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build parent_gt and k_true arrays of shape [N,S]."""
    kind = str(count_kind).lower()
    gs = group_size.astype(np.int32, copy=False)
    parent_gt = (hlt_mask & (gs > 1)).astype(np.float32) #parent_gt记录哪些contituents是parent(0，1)（after efficiency loss）
    if kind in ("missing", "delta", "deltan"):
        kk = np.maximum(gs - 1, 0) #kk记录每个contituent融合了几个其它contituent(不包括自己)（after efficiency loss）
    else:
        kk = np.maximum(gs, 0) #kk记录每个contituent融合了几个其它contituent(包括自己)（after efficiency loss）
    if max_k is not None:
        kk = np.minimum(kk, int(max_k))
    k_true = kk.astype(np.float32)
    k_true[~hlt_mask] = 0.0
    return parent_gt, k_true


class JetParentKDataset(Dataset):
    """One sample = one jet; supervise parentness and k on all tokens."""

    def __init__(
        self,
        jet_indices: Sequence[int],
        *,
        feat_hlt_std: np.ndarray,
        mask_hlt: np.ndarray,
        parent_gt: np.ndarray,
        k_true: np.ndarray,
    ):
        self.jet_idx = np.asarray(jet_indices, dtype=np.int64)
        self.hlt = torch.tensor(feat_hlt_std, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.parent_gt = torch.tensor(parent_gt, dtype=torch.float32)
        self.k_true = torch.tensor(k_true, dtype=torch.float32)

    def __len__(self):
        return int(self.jet_idx.shape[0])

    def __getitem__(self, i):
        j = int(self.jet_idx[i])
        return {
            "hlt": self.hlt[j],
            "mask_hlt": self.mask[j],
            "parent_gt": self.parent_gt[j],
            "k_true": self.k_true[j],
        }


class ParentRecoDataset(Dataset):
    """One sample = one merged parent token; supervise reconstruction ordered children."""

    def __init__(
        self,
        indices: Sequence[int],
        *,
        samples: Sequence[Tuple[int, int]],
        children: Sequence[np.ndarray],
        feat_hlt_std: np.ndarray,
        mask_hlt: np.ndarray,
        off_child_feat_std: np.ndarray,
        k_max: int,
        # Optionally supervise k on this parent token
        k_true_token: Optional[np.ndarray] = None,
        # Optionally add full token-level supervision for auxiliary losses in Stage2
        parent_gt: Optional[np.ndarray] = None,
        k_true: Optional[np.ndarray] = None,
    ):
        self.sel = np.asarray(indices, dtype=np.int64)
        self.samples = samples
        self.children = children
        self.hlt = torch.tensor(feat_hlt_std, dtype=torch.float32)
        self.mask = torch.tensor(mask_hlt, dtype=torch.bool)
        self.off_child_feat_std = torch.tensor(off_child_feat_std, dtype=torch.float32)
        self.k_max = int(k_max)
        self.k_true_token = None if k_true_token is None else torch.tensor(k_true_token, dtype=torch.float32)
        self.parent_gt = None if parent_gt is None else torch.tensor(parent_gt, dtype=torch.float32)
        self.k_true = None if k_true is None else torch.tensor(k_true, dtype=torch.float32)

    def __len__(self):
        return int(self.sel.shape[0])

    def __getitem__(self, i):
        sidx = int(self.sel[i])
        jet_idx, parent_idx = self.samples[sidx]
        ch = self.children[sidx]
        m = int(ch.shape[0])
        D = int(self.off_child_feat_std.shape[-1])
        tgt = torch.zeros((self.k_max, D), dtype=torch.float32)
        tgt_mask = torch.zeros((self.k_max,), dtype=torch.bool)
        tgt[:m] = self.off_child_feat_std[jet_idx, ch]
        tgt_mask[:m] = True

        out = {
            "hlt": self.hlt[jet_idx],
            "mask_hlt": self.mask[jet_idx],
            "parent_idx": torch.tensor(int(parent_idx), dtype=torch.long),
            "tgt": tgt,
            "tgt_mask": tgt_mask,
            "m_true": torch.tensor(float(m), dtype=torch.float32),
        }
        if self.k_true_token is not None:
            out["k_true_token"] = self.k_true_token[jet_idx, int(parent_idx)]
        if self.parent_gt is not None:
            out["parent_gt"] = self.parent_gt[jet_idx]
        if self.k_true is not None:
            out["k_true"] = self.k_true[jet_idx]
        return out


# -----------------------------
# Losses + training loops
# -----------------------------


def _masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, *, pos_weight: float = 1.0
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.new_tensor(0.0)
    logits_v = logits[mask]
    targets_v = targets[mask]
    pw = logits.new_tensor(float(pos_weight))
    return nn.BCEWithLogitsLoss(pos_weight=pw)(logits_v, targets_v)


def _masked_weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return pred.new_tensor(0.0)
    pred_v = pred[mask]
    target_v = target[mask]
    w_v = torch.clamp(w[mask], min=0.0)
    loss = F.smooth_l1_loss(pred_v, target_v, reduction="none", beta=float(delta))
    return (loss * w_v).sum() / (w_v.sum() + 1e-8)


def _masked_weighted_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted token-level cross-entropy.
    Args:
      logits: [B,S,K]
      target: [B,S] long in [0, K-1]
      mask: [B,S] bool
      w: [B,S] float weights (e.g. 1 for parents, neg_k_weight for non-parents)
    """
    if int(mask.sum().item()) == 0:
        return logits.new_tensor(0.0)
    lv = logits[mask]
    tv = target[mask]
    wv = torch.clamp(w[mask], min=0.0)
    ce = F.cross_entropy(lv, tv, reduction="none")  # [M]
    return (ce * wv).sum() / (wv.sum() + 1e-8)


@dataclass
class TrainCfgPK:
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 2
    patience: int = 6
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    thr_parent: float = 0.5
    w_parent: float = 1.0
    w_k: float = 1.0
    # Jet-level constraint: sum(k_pred on GT parents) ~= sum(k_true on GT parents)
    w_sum: float = 0.2
    neg_k_weight: float = 0.2


def _parent_sum_huber(
    k_pred: torch.Tensor,
    k_true: torch.Tensor,
    parent_gt: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Jet-level sum constraint on GT parents only:
      sum_i k_pred[i] ~= sum_i k_true[i], with i restricted to GT parent tokens.
    """
    gt_parent = (parent_gt > 0.5) & mask
    if int(gt_parent.sum().item()) == 0:
        return k_pred.new_tensor(0.0)
    # Only include jets that have at least one GT parent token
    jet_has = gt_parent.any(dim=1)
    if int(jet_has.sum().item()) == 0:
        return k_pred.new_tensor(0.0)
    w = gt_parent.to(dtype=k_pred.dtype)
    pred_sum = (k_pred * w).sum(dim=1)[jet_has]
    true_sum = (k_true * w).sum(dim=1)[jet_has]
    return F.smooth_l1_loss(pred_sum, true_sum, reduction="mean", beta=float(delta))


@dataclass
class TrainCfgReco:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    patience: int = 6
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    w_reco: float = 1.0
    w_k: float = 0.2  # optionally supervise k on parent token during reco
    freeze_encoder: bool = True  # freeze encoder + parent/k heads during Stage2
    # If not freezing encoder, you can use a smaller LR for encoder than decoder.
    enc_lr_mult: float = 0.1
    # Optional auxiliary parent/k losses during reco to prevent drift (set small).
    w_parent_aux: float = 0.0
    w_k_aux: float = 0.0
    w_sum_aux: float = 0.0
    neg_k_weight: float = 0.2  # used by k auxiliary loss on non-parents


def _cosine_schedule(ep: int, warmup: int, total: int) -> float:
    if ep < warmup:
        return float(ep + 1) / float(max(1, warmup))
    t = float(ep - warmup) / float(max(1, total - warmup))
    return float(0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t)))))


@torch.no_grad()
def eval_parent_k(model, loader, device: torch.device, *, pos_weight: float, cfg: TrainCfgPK) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    tot_parent = 0.0
    tot_k = 0.0
    tot_sum = 0.0
    nb = 0
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        y_parent = batch["parent_gt"].to(device)
        y_k = batch["k_true"].to(device)
        out = model(x, m, parent_idx=None)
        parent_loss = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight))
        tot_parent += float(parent_loss.item())
        w = torch.where(y_parent > 0.5, torch.ones_like(y_parent), y_parent.new_full(y_parent.shape, float(cfg.neg_k_weight)))
        if out.k_logits is not None:
            K = int(out.k_logits.shape[-1])
            y_ki = torch.clamp(y_k.round().to(torch.long), 0, K - 1)
            k_loss = _masked_weighted_ce(out.k_logits, y_ki, m, w)
        else:
            k_loss = _masked_weighted_huber(out.k_pred, y_k, m, w, delta=float(cfg.huber_delta))
        tot_k += float(k_loss.item())
        sum_loss = _parent_sum_huber(out.k_pred, y_k, y_parent, m, delta=float(cfg.huber_delta))
        tot_sum += float(sum_loss.item())

        prob = torch.sigmoid(out.parent_logit)
        pred = (prob > float(cfg.thr_parent)) & m
        gt = (y_parent > 0.5) & m
        tp += int((pred & gt).sum().item())
        fp += int((pred & (~gt)).sum().item())
        fn += int(((~pred) & gt).sum().item())
        nb += 1

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    return {
        "parent_loss": tot_parent / max(1, nb),
        "k_loss": tot_k / max(1, nb),
        "sum_loss": tot_sum / max(1, nb),
        "precision": float(prec),
        "recall": float(rec),
    }


def train_parent_k(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainCfgPK,
    *,
    pos_weight: float,
    ckpt_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    best = float("inf")
    best_state = None
    no_imp = 0
    hist: dict[str, list[float]] = {
        "train_total": [],
        "train_parent": [],
        "train_k": [],
        "train_sum": [],
        "val_parent": [],
        "val_k": [],
        "val_sum": [],
    }
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        lr_scale = _cosine_schedule(ep - 1, int(cfg.warmup_epochs), int(cfg.epochs))
        for g in opt.param_groups:
            g["lr"] = float(cfg.lr) * float(lr_scale)
        running_total = 0.0
        running_parent = 0.0
        running_k = 0.0
        running_sum = 0.0
        n = 0
        for batch in train_loader:
            x = batch["hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            y_parent = batch["parent_gt"].to(device)
            y_k = batch["k_true"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, m, parent_idx=None)
            parent_loss = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight)) #加权BCE loss作为parent_loss
            w = torch.where(y_parent > 0.5, torch.ones_like(y_parent), y_parent.new_full(y_parent.shape, float(cfg.neg_k_weight)))
            if out.k_logits is not None:
                K = int(out.k_logits.shape[-1])
                y_ki = torch.clamp(y_k.round().to(torch.long), 0, K - 1)
                k_loss = _masked_weighted_ce(out.k_logits, y_ki, m, w)
            else:
                k_loss = _masked_weighted_huber(out.k_pred, y_k, m, w, delta=float(cfg.huber_delta)) #加权Huber loss作为k_loss，注意neg_k_weight为0.2表示负样本的权重不如正样本重要，言下之意，我们并不关心不是parent的k值
            sum_loss = _parent_sum_huber(out.k_pred, y_k, y_parent, m, delta=float(cfg.huber_delta))
            loss = float(cfg.w_parent) * parent_loss + float(cfg.w_k) * k_loss + float(cfg.w_sum) * sum_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            running_total += float(loss.item())
            running_parent += float(parent_loss.item())
            running_k += float(k_loss.item())
            running_sum += float(sum_loss.item())
            n += 1
        val = eval_parent_k(model, val_loader, device, pos_weight=float(pos_weight), cfg=cfg)
        val_scalar = float(val["parent_loss"] + val["k_loss"] + float(cfg.w_sum) * val.get("sum_loss", 0.0))
        if val_scalar < best - 1e-6:
            best = val_scalar
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, extra={"epoch": ep, "best": best, "val": val})
        else:
            no_imp += 1
        hist["train_total"].append(float(running_total / max(1, n)))
        hist["train_parent"].append(float(running_parent / max(1, n)))
        hist["train_k"].append(float(running_k / max(1, n)))
        hist["train_sum"].append(float(running_sum / max(1, n)))
        hist["val_parent"].append(float(val["parent_loss"]))
        hist["val_k"].append(float(val["k_loss"]))
        hist["val_sum"].append(float(val.get("sum_loss", 0.0)))
        print(
            f"[Parent+K] Ep {ep:03d}: train_total={hist['train_total'][-1]:.4f} "
            f"(p={hist['train_parent'][-1]:.4f}, k={hist['train_k'][-1]:.4f}, sum={hist['train_sum'][-1]:.4f}) "
            f"val_p={val['parent_loss']:.4f} val_k={val['k_loss']:.4f} val_sum={val.get('sum_loss', 0.0):.4f} "
            f"prec={val['precision']:.3f} rec={val['recall']:.3f} no_imp={no_imp}"
        )
        if no_imp >= int(cfg.patience):
            print("[Parent+K] Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best": best, "history": hist}


@torch.no_grad()
def eval_reco(model, loader, device: torch.device, *, cfg: TrainCfgReco) -> dict[str, float]:
    model.eval()
    tot = 0.0
    tot_reco = 0.0
    tot_k_token = 0.0
    tot_parent_aux = 0.0
    tot_k_aux = 0.0
    tot_sum_aux = 0.0
    nb = 0
    mae_m = 0.0
    for batch in loader:
        x = batch["hlt"].to(device)
        m = batch["mask_hlt"].to(device)
        parent_idx = batch["parent_idx"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)
        out = model(x, m, parent_idx=parent_idx)
        pred = out.child_feat
        assert pred is not None
        # Ordered: regress only on true slots
        w_mask = tgt_mask.unsqueeze(-1).to(pred.dtype)
        loss_reco = F.smooth_l1_loss(pred * w_mask, tgt * w_mask, reduction="sum", beta=float(cfg.huber_delta))
        denom = w_mask.sum().clamp_min(1.0)
        loss_reco = loss_reco / denom
        loss = float(cfg.w_reco) * loss_reco
        tot_reco += float(loss_reco.item())

        # Optional k supervision on parent token (if present)
        if "k_true_token" in batch:
            k_true_token = batch["k_true_token"].to(device)
            bi = torch.arange(int(x.shape[0]), device=device)
            if out.k_logits is not None:
                logits_tok = out.k_logits[bi, parent_idx]  # [B,K]
                K = int(logits_tok.shape[-1])
                y_ki = torch.clamp(k_true_token.round().to(torch.long), 0, K - 1)
                loss_k = F.cross_entropy(logits_tok, y_ki, reduction="mean")
                loss = loss + float(cfg.w_k) * loss_k
            else:
                k_pred_token = out.k_pred[bi, parent_idx]
                loss_k = F.smooth_l1_loss(k_pred_token, k_true_token, reduction="mean", beta=float(cfg.huber_delta))
                loss = loss + float(cfg.w_k) * loss_k
            tot_k_token += float(loss_k.item())

        # Optional auxiliary losses (if batch provides full jet targets)
        if float(getattr(cfg, "w_parent_aux", 0.0)) > 0.0 and ("parent_gt" in batch):
            y_parent = batch["parent_gt"].to(device)
            loss_parent_aux = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=1.0)
            loss = loss + float(cfg.w_parent_aux) * loss_parent_aux
            tot_parent_aux += float(loss_parent_aux.item())
        if float(getattr(cfg, "w_k_aux", 0.0)) > 0.0 and ("k_true" in batch) and ("parent_gt" in batch):
            y_parent = batch["parent_gt"].to(device)
            y_k = batch["k_true"].to(device)
            w = torch.where(y_parent > 0.5, torch.ones_like(y_parent), y_parent.new_full(y_parent.shape, float(cfg.neg_k_weight)))
            if out.k_logits is not None:
                K = int(out.k_logits.shape[-1])
                y_ki = torch.clamp(y_k.round().to(torch.long), 0, K - 1)
                loss_k_aux = _masked_weighted_ce(out.k_logits, y_ki, m, w)
            else:
                loss_k_aux = _masked_weighted_huber(out.k_pred, y_k, m, w, delta=float(cfg.huber_delta))
            loss = loss + float(cfg.w_k_aux) * loss_k_aux
            tot_k_aux += float(loss_k_aux.item())
        if float(getattr(cfg, "w_sum_aux", 0.0)) > 0.0 and ("k_true" in batch) and ("parent_gt" in batch):
            y_parent = batch["parent_gt"].to(device)
            y_k = batch["k_true"].to(device)
            loss_sum_aux = _parent_sum_huber(out.k_pred, y_k, y_parent, m, delta=float(cfg.huber_delta))
            loss = loss + float(cfg.w_sum_aux) * loss_sum_aux
            tot_sum_aux += float(loss_sum_aux.item())

        tot += float(loss.item())
        nb += 1

        # k as proxy: compare round(k_pred_token) to m_true
        if "k_true_token" in batch:
            m_true = batch["m_true"].to(device)
            if out.k_logits is not None:
                k_pred_token = out.k_pred[bi, parent_idx]
            mae_m += float((k_pred_token - m_true).abs().mean().item())
    return {
        "loss": tot / max(1, nb),
        "loss_reco": tot_reco / max(1, nb),
        "loss_k_token": tot_k_token / max(1, nb),
        "loss_parent_aux": tot_parent_aux / max(1, nb),
        "loss_k_aux": tot_k_aux / max(1, nb),
        "loss_sum_aux": tot_sum_aux / max(1, nb),
        "mae_k_vs_m": mae_m / max(1, nb) if nb > 0 else float("nan"),
    }


def train_reco_teacher_forced(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainCfgReco,
    *,
    pos_weight: float = 1.0,
    ckpt_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    # Two options:
    # - freeze_encoder=True: freeze encoder + parent/k heads, train decoder only
    # - freeze_encoder=False: train full model but use low LR for encoder side and high LR for decoder side
    orig_flags = [(p, bool(p.requires_grad)) for p in model.parameters()]

    def _set_requires_grad(params: list[torch.nn.Parameter], flag: bool) -> None:
        for p in params:
            p.requires_grad = bool(flag)

    # Split params into encoder-side and decoder-side
    enc_params: list[torch.nn.Parameter] = []
    dec_params: list[torch.nn.Parameter] = []
    for name in ["input_proj", "encoder", "parent_head", "k_head"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                enc_params += [p for p in mod.parameters()]
    # Decoder-side (includes query_embed parameter)
    if hasattr(model, "query_embed") and isinstance(getattr(model, "query_embed"), torch.nn.Parameter):
        dec_params.append(getattr(model, "query_embed"))
    for name in ["decoder", "cond_proj", "child_head"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                dec_params += [p for p in mod.parameters()]

    # De-duplicate while preserving order
    seen: set[int] = set()
    enc_params_u: list[torch.nn.Parameter] = []
    for p in enc_params:
        if id(p) not in seen:
            enc_params_u.append(p)
            seen.add(id(p))
    dec_params_u: list[torch.nn.Parameter] = []
    for p in dec_params:
        if id(p) not in seen:
            dec_params_u.append(p)
            seen.add(id(p))

    frozen_mods: list[nn.Module] = []
    if bool(getattr(cfg, "freeze_encoder", True)):
        _set_requires_grad(enc_params_u, False)
        for name in ["input_proj", "encoder", "parent_head", "k_head"]:
            if hasattr(model, name) and isinstance(getattr(model, name), nn.Module):
                frozen_mods.append(getattr(model, name))
        _set_requires_grad(dec_params_u, True)
        param_groups = [
            {"params": [p for p in dec_params_u if p.requires_grad], "base_lr": float(cfg.lr)},
        ]
    else:
        _set_requires_grad(enc_params_u, True)
        _set_requires_grad(dec_params_u, True)
        enc_lr = float(cfg.lr) * float(getattr(cfg, "enc_lr_mult", 0.1))
        param_groups = [
            {"params": [p for p in enc_params_u if p.requires_grad], "base_lr": float(enc_lr)},
            {"params": [p for p in dec_params_u if p.requires_grad], "base_lr": float(cfg.lr)},
        ]

    if sum(len(g["params"]) for g in param_groups) == 0:
        raise RuntimeError("No trainable parameters found for reco training (check freeze/lr settings).")
    opt = torch.optim.AdamW(param_groups, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    best = float("inf")
    best_state = None
    no_imp = 0
    hist: dict[str, list[float]] = {
        "train_total": [],
        "train_reco": [],
        "train_k_token": [],
        "train_parent_aux": [],
        "train_k_aux": [],
        "train_sum_aux": [],
        "val_total": [],
        "val_reco": [],
        "val_k_token": [],
        "val_parent_aux": [],
        "val_k_aux": [],
        "val_sum_aux": [],
    }
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        for fm in frozen_mods:
            fm.eval()
        lr_scale = _cosine_schedule(ep - 1, int(cfg.warmup_epochs), int(cfg.epochs))
        for g in opt.param_groups:
            g["lr"] = float(g.get("base_lr", float(cfg.lr))) * float(lr_scale)
        running_total = 0.0
        running_reco = 0.0
        running_k_token = 0.0
        running_parent_aux = 0.0
        running_k_aux = 0.0
        running_sum_aux = 0.0
        n = 0
        for batch in train_loader:
            x = batch["hlt"].to(device)
            m = batch["mask_hlt"].to(device)
            parent_idx = batch["parent_idx"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, m, parent_idx=parent_idx)
            pred = out.child_feat
            assert pred is not None
            w_mask = tgt_mask.unsqueeze(-1).to(pred.dtype)
            loss_reco = F.smooth_l1_loss(pred * w_mask, tgt * w_mask, reduction="sum", beta=float(cfg.huber_delta))
            denom = w_mask.sum().clamp_min(1.0)
            loss_reco = loss_reco / denom
            loss = float(cfg.w_reco) * loss_reco
            running_reco += float(loss_reco.item())

            if "k_true_token" in batch:
                k_true_token = batch["k_true_token"].to(device)
                bi = torch.arange(int(x.shape[0]), device=device)
                if out.k_logits is not None:
                    logits_tok = out.k_logits[bi, parent_idx]  # [B,K]
                    K = int(logits_tok.shape[-1])
                    y_ki = torch.clamp(k_true_token.round().to(torch.long), 0, K - 1)
                    loss_k = F.cross_entropy(logits_tok, y_ki, reduction="mean")
                    loss = loss + float(cfg.w_k) * loss_k
                else:
                    k_pred_token = out.k_pred[bi, parent_idx]
                    loss_k = F.smooth_l1_loss(k_pred_token, k_true_token, reduction="mean", beta=float(cfg.huber_delta))
                    loss = loss + float(cfg.w_k) * loss_k
                running_k_token += float(loss_k.item())

            # Optional auxiliary losses to prevent parent/k drift during Stage2
            if float(getattr(cfg, "w_parent_aux", 0.0)) > 0.0 and ("parent_gt" in batch):
                y_parent = batch["parent_gt"].to(device)
                loss_parent_aux = _masked_bce_with_logits(out.parent_logit, y_parent, m, pos_weight=float(pos_weight))
                loss = loss + float(cfg.w_parent_aux) * loss_parent_aux
                running_parent_aux += float(loss_parent_aux.item())
            if float(getattr(cfg, "w_k_aux", 0.0)) > 0.0 and ("k_true" in batch) and ("parent_gt" in batch):
                y_parent = batch["parent_gt"].to(device)
                y_k = batch["k_true"].to(device)
                w = torch.where(y_parent > 0.5, torch.ones_like(y_parent), y_parent.new_full(y_parent.shape, float(cfg.neg_k_weight)))
                if out.k_logits is not None:
                    K = int(out.k_logits.shape[-1])
                    y_ki = torch.clamp(y_k.round().to(torch.long), 0, K - 1)
                    loss_k_aux = _masked_weighted_ce(out.k_logits, y_ki, m, w)
                else:
                    loss_k_aux = _masked_weighted_huber(out.k_pred, y_k, m, w, delta=float(cfg.huber_delta))
                loss = loss + float(cfg.w_k_aux) * loss_k_aux
                running_k_aux += float(loss_k_aux.item())
            if float(getattr(cfg, "w_sum_aux", 0.0)) > 0.0 and ("k_true" in batch) and ("parent_gt" in batch):
                y_parent = batch["parent_gt"].to(device)
                y_k = batch["k_true"].to(device)
                loss_sum_aux = _parent_sum_huber(out.k_pred, y_k, y_parent, m, delta=float(cfg.huber_delta))
                loss = loss + float(cfg.w_sum_aux) * loss_sum_aux
                running_sum_aux += float(loss_sum_aux.item())

            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.grad_clip))
            opt.step()
            running_total += float(loss.item())
            n += 1

        val = eval_reco(model, val_loader, device, cfg=cfg)
        v = float(val["loss"])
        if v < best - 1e-6:
            best = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, extra={"epoch": ep, "best": best, "val": val})
        else:
            no_imp += 1
        hist["train_total"].append(float(running_total / max(1, n)))
        hist["train_reco"].append(float(running_reco / max(1, n)))
        hist["train_k_token"].append(float(running_k_token / max(1, n)))
        hist["train_parent_aux"].append(float(running_parent_aux / max(1, n)))
        hist["train_k_aux"].append(float(running_k_aux / max(1, n)))
        hist["train_sum_aux"].append(float(running_sum_aux / max(1, n)))

        hist["val_total"].append(float(v))
        hist["val_reco"].append(float(val.get("loss_reco", 0.0)))
        hist["val_k_token"].append(float(val.get("loss_k_token", 0.0)))
        hist["val_parent_aux"].append(float(val.get("loss_parent_aux", 0.0)))
        hist["val_k_aux"].append(float(val.get("loss_k_aux", 0.0)))
        hist["val_sum_aux"].append(float(val.get("loss_sum_aux", 0.0)))

        print(
            f"[Reco/TF] Ep {ep:03d}: train_total={hist['train_total'][-1]:.4f} "
            f"(reco={hist['train_reco'][-1]:.4f}, k_tok={hist['train_k_token'][-1]:.4f}, "
            f"p_aux={hist['train_parent_aux'][-1]:.4f}, k_aux={hist['train_k_aux'][-1]:.4f}, sum_aux={hist['train_sum_aux'][-1]:.4f}) "
            f"val_total={v:.4f} val_reco={hist['val_reco'][-1]:.4f} val_k_tok={hist['val_k_token'][-1]:.4f} "
            f"val_p_aux={hist['val_parent_aux'][-1]:.4f} val_k_aux={hist['val_k_aux'][-1]:.4f} val_sum_aux={hist['val_sum_aux'][-1]:.4f} "
            f"no_imp={no_imp}"
        )
        if no_imp >= int(cfg.patience):
            print("[Reco/TF] Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # Restore original requires_grad flags for safety (so later stages can fine-tune if desired).
    for p, f in orig_flags:
        p.requires_grad = bool(f)
    return {"best": best, "history": hist}


# -----------------------------
# Inference: build unmerged view (Ordered + k_pred)
# -----------------------------


def pack_topN_raw(tokens_raw: np.ndarray, max_particles: int) -> tuple[np.ndarray, np.ndarray]:
    if tokens_raw.shape[0] == 0:
        out = np.zeros((max_particles, 4), dtype=np.float32)
        mask = np.zeros((max_particles,), dtype=np.bool_)
        return out, mask
    pts = tokens_raw[:, 0]
    order = np.argsort(-pts)
    tokens_raw = tokens_raw[order]
    m = min(int(max_particles), int(tokens_raw.shape[0]))
    out = np.zeros((max_particles, 4), dtype=np.float32)
    mask = np.zeros((max_particles,), dtype=np.bool_)
    out[:m] = tokens_raw[:m]
    mask[:m] = True
    return out, mask


@torch.no_grad()
def build_unmerged_view_ordered(
    model,
    device: torch.device,
    *,
    hlt_raw: np.ndarray,
    hlt_mask: np.ndarray,
    hlt_feat_std: np.ndarray,
    hlt_axis: np.ndarray,  # [N,4] => (eta,phi,pt,E)
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    max_particles: int,
    thr_parent: float = 0.7,
    max_split_parents: Optional[int] = 20,
    max_k_per_parent: Optional[int] = 8,
    count_kind: str = "missing",
    k_infer_mode: str = "expectation",  # "expectation" | "argmax" (only used when k_mode='class')
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build unmerged raw+feats view by:
      - selecting parent tokens by parentness
      - predicting k per selected parent (ordered slots)
      - replacing those parent tokens with the first k children predicted
      - packing top-N by pt
    """
    model.eval()
    N, S, _ = hlt_raw.shape
    out_raw = np.zeros((N, max_particles, 4), dtype=np.float32)
    out_mask = np.zeros((N, max_particles), dtype=np.bool_)

    bs = int(max(1, batch_size))
    for s0 in range(0, N, bs):
        s1 = min(N, s0 + bs)
        # Batch parentness inference (fast path)
        xb = torch.tensor(hlt_feat_std[s0:s1], dtype=torch.float32, device=device)
        mb = torch.tensor(hlt_mask[s0:s1], dtype=torch.bool, device=device)
        outb = model(xb, mb, parent_idx=None)
        prob_b = torch.sigmoid(outb.parent_logit).detach().cpu().numpy()  # [B,S]

        for bi, j in enumerate(range(s0, s1)):
            prob = prob_b[bi]
            valid = np.where(hlt_mask[j])[0]
            parents = np.asarray([i for i in valid if prob[int(i)] > float(thr_parent)], dtype=np.int64)
            if max_split_parents is not None and parents.shape[0] > int(max_split_parents):
                order = np.argsort(-prob[parents])[: int(max_split_parents)]
                parents = parents[order]

            keep = hlt_mask[j].copy()
            tokens: list[np.ndarray] = []
            for p in parents:
                keep[int(p)] = False
            keep_idx = np.where(keep)[0]
            if keep_idx.size > 0:
                tokens.append(hlt_raw[j, keep_idx])

            # Teacher-forced-like conditional decode per selected parent (still per-jet, but avoids re-tensor conversion)
            if parents.size > 0:
                P = int(parents.shape[0])
                xj = xb[bi : bi + 1]  # [1,S,D]
                mj = mb[bi : bi + 1]  # [1,S]
                xP = xj.repeat(P, 1, 1)
                mP = mj.repeat(P, 1)
                pidx = torch.tensor(parents, dtype=torch.long, device=device)
                outP = model(xP, mP, parent_idx=pidx)
                child = outP.child_feat.detach().cpu().numpy()  # [P,K,D] in standardized feat space
                # k_pred is expectation if k_mode='class', regression otherwise
                k_pred_tok = outP.k_pred.detach().cpu().numpy()  # [P,S]
                k_logits_tok = outP.k_logits.detach().cpu().numpy() if outP.k_logits is not None else None  # [P,S,K]
                for pi in range(P):
                    # 这里决定推理用哪种 k：
                    # - expectation: 用 E[k]（可当作连续值，再 round）
                    # - argmax: 用 argmax(P(k=j)) 得到离散 k
                    if k_logits_tok is not None and str(k_infer_mode).lower() == "argmax":
                        k_val = int(np.argmax(k_logits_tok[pi, int(parents[pi])], axis=-1))
                    else:
                        k_val = float(k_pred_tok[pi, int(parents[pi])])
                    if str(count_kind).lower() in ("missing", "delta", "deltan"):
                        k_child = int(max(1, int(np.round(k_val)) + 1))
                    else:
                        k_child = int(max(0, int(np.round(k_val))))
                    if max_k_per_parent is not None:
                        k_child = min(int(k_child), int(max_k_per_parent))
                    k_child = max(0, min(int(k_child), child.shape[1]))
                    if k_child <= 0:
                        continue
                    feats = child[pi, :k_child]  # [k,D] (std)
                    feats = feats * feat_stds[None, :] + feat_means[None, :]  # unstandardize
                    je = np.asarray([hlt_axis[j, 0]], dtype=np.float32)
                    jp = np.asarray([hlt_axis[j, 1]], dtype=np.float32)
                    raw_child = feats_to_raw(feats[None, ...], je, jp)[0]
                    tokens.append(raw_child)

            tokens_raw = np.concatenate(tokens, axis=0) if tokens else np.zeros((0, 4), dtype=np.float32)
            packed, pmask = pack_topN_raw(tokens_raw, max_particles=int(max_particles))
            out_raw[j] = packed
            out_mask[j] = pmask

    # Build standardized features in the same feature space for downstream (optional)
    D = int(np.asarray(feat_means).shape[0])
    kind = "7d" if D >= 7 else "4d"
    feats = raw_to_feats(out_raw, out_mask, hlt_axis, kind=kind)
    feats_std = standardize_tokens(feats, out_mask, feat_means, feat_stds, clip=10.0)
    return out_raw, out_mask, feats_std


# -----------------------------
# Downstream tagger datasets + KD utilities
# -----------------------------


class JetTaggerDataset(Dataset):
    """
    One sample = one jet for classification.

    Provides OFF/HLT/UNM views so the same DataLoader can be reused across models.
    """

    def __init__(
        self,
        jet_indices: Sequence[int],
        *,
        y: np.ndarray,
        w: Optional[np.ndarray],
        off_feat_std: np.ndarray,
        off_mask: np.ndarray,
        hlt_feat_std: np.ndarray,
        hlt_mask: np.ndarray,
        unm_feat_std: Optional[np.ndarray] = None,
        unm_mask: Optional[np.ndarray] = None,
    ):
        self.jet_idx = np.asarray(jet_indices, dtype=np.int64)
        self.y = torch.tensor(y, dtype=torch.float32)
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        self.off = torch.tensor(off_feat_std, dtype=torch.float32)
        self.off_mask = torch.tensor(off_mask, dtype=torch.bool)
        self.hlt = torch.tensor(hlt_feat_std, dtype=torch.float32)
        self.hlt_mask = torch.tensor(hlt_mask, dtype=torch.bool)
        self.unm = None if unm_feat_std is None else torch.tensor(unm_feat_std, dtype=torch.float32)
        self.unm_mask = None if unm_mask is None else torch.tensor(unm_mask, dtype=torch.bool)

    def __len__(self):
        return int(self.jet_idx.shape[0])

    def __getitem__(self, i):
        j = int(self.jet_idx[i])
        out = {
            "y": self.y[j],
            "w": self.w[j],
            "off": self.off[j],
            "off_mask": self.off_mask[j],
            "hlt": self.hlt[j],
            "hlt_mask": self.hlt_mask[j],
        }
        if self.unm is not None and self.unm_mask is not None:
            out["unm"] = self.unm[j]
            out["unm_mask"] = self.unm_mask[j]
        return out


def kd_kl_loss_binary(student_logit: torch.Tensor, teacher_logit: torch.Tensor, T: float) -> torch.Tensor:
    """KL between 2-class softmax distributions built from a single logit."""
    s2 = torch.stack([torch.zeros_like(student_logit), student_logit], dim=-1) / float(T)
    t2 = torch.stack([torch.zeros_like(teacher_logit), teacher_logit], dim=-1) / float(T)
    logp_s = F.log_softmax(s2, dim=-1)
    p_t = F.softmax(t2, dim=-1)
    return F.kl_div(logp_s, p_t, reduction="batchmean") * (float(T) ** 2)


def attn_loss(s_attn: torch.Tensor, t_attn: torch.Tensor, s_mask: torch.Tensor, t_mask: torch.Tensor) -> torch.Tensor:
    """Attention distillation loss (entropy + max), expects [B,S] attention weights."""
    eps = 1e-8
    if int(s_attn.shape[-1]) != int(t_attn.shape[-1]):
        L = int(min(int(s_attn.shape[-1]), int(t_attn.shape[-1])))
        s_attn = s_attn[:, :L]
        t_attn = t_attn[:, :L]
        s_mask = s_mask[:, :L]
        t_mask = t_mask[:, :L]
    s_valid = s_attn * s_mask.to(dtype=s_attn.dtype)
    t_valid = t_attn * t_mask.to(dtype=t_attn.dtype)
    s_ent = -(s_valid * torch.log(s_valid + eps)).sum(dim=1)
    t_ent = -(t_valid * torch.log(t_valid + eps)).sum(dim=1)
    return F.mse_loss(s_ent, t_ent) + F.mse_loss(s_valid.max(dim=1)[0], t_valid.max(dim=1)[0])


@torch.no_grad()
def eval_auc_logits(model, loader, device: torch.device, *, kind: str) -> float:
    """
    kind:
      - 'teacher'  : model(off, off_mask)
      - 'hlt'      : model(hlt, hlt_mask)
      - 'dual'     : model(hlt, hlt_mask, unm, unm_mask)
    """
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        y = batch["y"].to(device)
        if kind == "teacher":
            logit = model(batch["off"].to(device), batch["off_mask"].to(device))
        elif kind == "hlt":
            logit = model(batch["hlt"].to(device), batch["hlt_mask"].to(device))
        elif kind == "dual":
            logit = model(
                batch["hlt"].to(device),
                batch["hlt_mask"].to(device),
                batch["unm"].to(device),
                batch["unm_mask"].to(device),
            )
        else:
            raise ValueError(kind)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0)
    p_np = np.concatenate(ps, axis=0)
    return float(roc_auc_score(y_np, p_np))


@torch.no_grad()
def collect_probs_logits(
    model,
    loader,
    device: torch.device,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p, y) numpy arrays for ROC plotting."""
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        y = batch["y"].to(device)
        if kind == "teacher":
            logit = model(batch["off"].to(device), batch["off_mask"].to(device))
        elif kind == "hlt":
            logit = model(batch["hlt"].to(device), batch["hlt_mask"].to(device))
        elif kind == "dual":
            logit = model(
                batch["hlt"].to(device),
                batch["hlt_mask"].to(device),
                batch["unm"].to(device),
                batch["unm_mask"].to(device),
            )
        else:
            raise ValueError(kind)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0).astype(np.float64)
    p_np = np.concatenate(ps, axis=0).astype(np.float64)
    return p_np, y_np


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    *,
    title: str = "ROC (test)",
    save_path: str | Path | None = None,
    dpi: int = 160,
):
    """curves[name] = (fpr, tpr, auc)."""
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(6.8, 6.0))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p.as_posix(), dpi=int(dpi), bbox_inches="tight")
        print(f"Saved figure: {p}")
    plt.show()


def _set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = float(lr)


def train_teacher(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            x = batch["off"].to(device)
            m = batch["off_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(model, val_loader, device, kind="teacher")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[Teacher] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[Teacher] Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, {"best_auc": best}


def train_hlt_baseline(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            x = batch["hlt"].to(device)
            m = batch["hlt_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(model, val_loader, device, kind="hlt")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[HLT] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[HLT] Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, {"best_auc": best}


def train_hlt_kd(
    student,
    teacher,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    temperature: float = 3.0,
    alpha_kd: float = 0.5,
    alpha_attn: float = 0.0,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    teacher.eval()
    for ep in range(1, int(epochs) + 1):
        student.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            hlt = batch["hlt"].to(device)
            hlt_m = batch["hlt_mask"].to(device)
            off = batch["off"].to(device)
            off_m = batch["off_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if float(alpha_attn) > 0.0:
                s_logit, s_attn = student(hlt, hlt_m, return_attention=True)
                ce = F.binary_cross_entropy_with_logits(s_logit, y)
                with torch.no_grad():
                    t_logit, t_attn = teacher(off, off_m, return_attention=True)
                kd = kd_kl_loss_binary(s_logit, t_logit, T=float(temperature))
                la = attn_loss(s_attn, t_attn, hlt_m, off_m)
                loss = (1.0 - float(alpha_kd)) * ce + float(alpha_kd) * kd + float(alpha_attn) * la
            else:
                s_logit = student(hlt, hlt_m)
                ce = F.binary_cross_entropy_with_logits(s_logit, y)
                with torch.no_grad():
                    t_logit = teacher(off, off_m)
                kd = kd_kl_loss_binary(s_logit, t_logit, T=float(temperature))
                loss = (1.0 - float(alpha_kd)) * ce + float(alpha_kd) * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(student, val_loader, device, kind="hlt")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[HLT+KD] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[HLT+KD] Early stopping.")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    return best, {"best_auc": best}


def train_dual_student_kd(
    student,
    teacher,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    temperature: float = 3.0,
    alpha_kd: float = 0.5,
    alpha_attn: float = 0.0,
    grad_clip: float = 1.0,
) -> tuple[float, dict]:
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = -1.0
    best_state = None
    no_imp = 0
    teacher.eval()
    for ep in range(1, int(epochs) + 1):
        student.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            hlt = batch["hlt"].to(device)
            hlt_m = batch["hlt_mask"].to(device)
            unm = batch["unm"].to(device)
            unm_m = batch["unm_mask"].to(device)
            off = batch["off"].to(device)
            off_m = batch["off_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if float(alpha_attn) > 0.0:
                s_logit, s_attn = student(hlt, hlt_m, unm, unm_m, return_attention=True)
            else:
                s_logit = student(hlt, hlt_m, unm, unm_m)
            ce = F.binary_cross_entropy_with_logits(s_logit, y)
            with torch.no_grad():
                if float(alpha_attn) > 0.0:
                    t_logit, t_attn = teacher(off, off_m, return_attention=True)
                else:
                    t_logit = teacher(off, off_m)
            kd = kd_kl_loss_binary(s_logit, t_logit, T=float(temperature))
            loss = (1.0 - float(alpha_kd)) * ce + float(alpha_kd) * kd
            if float(alpha_attn) > 0.0:
                loss = loss + float(alpha_attn) * attn_loss(s_attn, t_attn, hlt_m, off_m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(grad_clip))
            opt.step()
        val_auc = eval_auc_logits(student, val_loader, device, kind="dual")
        improved = val_auc > best + 1e-4
        if improved:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"[Dual+KD] Ep {ep:03d}: val_auc={val_auc:.5f} best={best:.5f} no_imp={no_imp}")
        if no_imp >= int(patience):
            print("[Dual+KD] Early stopping.")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    return best, {"best_auc": best}

