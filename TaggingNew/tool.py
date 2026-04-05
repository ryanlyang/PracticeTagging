from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


@dataclass
class CommonSmearCfg:
    """常用的 constituent smear 配置。"""

    pt_threshold_offline: float = 0.5
    pt_threshold_hlt: float = 0.5
    pt_resolution: float = 0.10
    eta_resolution: float = 0.03
    phi_resolution: float = 0.03


def wrap_dphi_np(dphi: np.ndarray) -> np.ndarray:
    """将角度差 wrap 到 (-pi, pi]。"""
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def _print_distribution_stats(
    name: str,
    values: np.ndarray,
    *,
    percentiles: Sequence[float],
) -> None:
    """打印一维数组的基础统计信息。"""
    print(f"\n[{name}] used_values={int(values.size):,}")
    if values.size == 0:
        print("No values after masking.")
        return

    print(
        f"  min={values.min():.6g} max={values.max():.6g} "
        f"mean={values.mean():.6g} std={values.std():.6g}"
    )
    ps = np.percentile(values, percentiles)
    for p, v in zip(percentiles, ps):
        print(f"  p{p:>5}={v:.6g}")


def inspect_dist_h5(
    h5_file: h5py.File,
    key: str,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
    bins: int = 120,
    logy: bool = False,
    mask_zero: bool = True,
    value_clip: tuple[float, float] | None = None,
    transform: str | None = None,
    percentiles: Sequence[float] = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9),
) -> np.ndarray:
    """检查 H5 数据集的一维分布。"""
    ds = h5_file[key]
    n_sel = int(min(max(1, n_jets), int(ds.shape[0])))

    if ds.ndim == 1:
        x = np.asarray(ds[:n_sel])
    elif ds.ndim == 2:
        s = int(min(int(ds.shape[1]), max_particles)) if max_particles is not None else int(ds.shape[1])
        x = np.asarray(ds[:n_sel, :s])
    else:
        raise ValueError(f"Unsupported ndim={ds.ndim} for key={key}")

    x = x.reshape(-1).astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if mask_zero:
        x = x[x != 0]

    if value_clip is not None:
        lo, hi = float(value_clip[0]), float(value_clip[1])
        x = np.clip(x, lo, hi)

    if transform is not None:
        t = str(transform).lower()
        if t == "log10":
            x = np.log10(np.clip(x, 1e-12, None))
        elif t in ("ln", "log"):
            x = np.log(np.clip(x, 1e-12, None))
        else:
            raise ValueError(f"Unknown transform={transform!r}")

    print(f"\n[{key}] used_jets={n_sel} raw_shape={tuple(ds.shape)}")
    _print_distribution_stats(key, x, percentiles=percentiles)
    if x.size == 0:
        return x

    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=int(bins), alpha=0.85)
    plt.title(f"{key} distribution")
    plt.xlabel(f"{key}" + (f" ({transform})" if transform else ""))
    plt.ylabel("counts")
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    return x


def _load_constituent_arrays(
    h5_file: h5py.File,
    *,
    n_jets: int,
    max_particles: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取前 N 个 jet 的 constituent 四个物理量，并构建有效 mask。"""
    pt_ds = h5_file["fjet_clus_pt"]
    eta_ds = h5_file["fjet_clus_eta"]
    phi_ds = h5_file["fjet_clus_phi"]
    e_ds = h5_file["fjet_clus_E"]

    n_sel = int(min(max(1, n_jets), int(pt_ds.shape[0])))
    s = int(min(int(pt_ds.shape[1]), max_particles)) if max_particles is not None else int(pt_ds.shape[1])

    pt = np.asarray(pt_ds[:n_sel, :s], dtype=np.float64)
    eta = np.asarray(eta_ds[:n_sel, :s], dtype=np.float64)
    phi = np.asarray(phi_ds[:n_sel, :s], dtype=np.float64)
    energy = np.asarray(e_ds[:n_sel, :s], dtype=np.float64)

    mask = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & np.isfinite(energy) & (pt > 0.0)
    return pt, eta, phi, energy, mask


def _apply_constituent_smear(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray,
    *,
    cfg: CommonSmearCfg,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对 constituent 四动量施加常用 smear 效果。"""
    rs = np.random.RandomState(int(seed))

    pt_out = np.asarray(pt, dtype=np.float64).copy()
    eta_out = np.asarray(eta, dtype=np.float64).copy()
    phi_out = np.asarray(phi, dtype=np.float64).copy()
    energy_out = np.asarray(energy, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool).copy()

    pt_thr_off = float(cfg.pt_threshold_offline)
    valid &= pt_out >= pt_thr_off
    pt_out[~valid] = 0.0
    eta_out[~valid] = 0.0
    phi_out[~valid] = 0.0
    energy_out[~valid] = 0.0

    pt_thr_hlt = float(cfg.pt_threshold_hlt)
    below = (pt_out < pt_thr_hlt) & valid
    valid[below] = False
    pt_out[~valid] = 0.0
    eta_out[~valid] = 0.0
    phi_out[~valid] = 0.0
    energy_out[~valid] = 0.0

    shape = pt_out.shape
    pt_noise = np.clip(rs.normal(1.0, float(cfg.pt_resolution), size=shape), 0.5, 1.5)
    eta_noise = rs.normal(0.0, float(cfg.eta_resolution), size=shape)
    phi_noise = rs.normal(0.0, float(cfg.phi_resolution), size=shape)

    pt_out = np.where(valid, pt_out * pt_noise, 0.0)
    eta_out = np.where(valid, np.clip(eta_out + eta_noise, -5.0, 5.0), 0.0)
    phi_out = np.where(valid, wrap_dphi_np(phi_out + phi_noise), 0.0)
    energy_out = np.where(valid, pt_out * np.cosh(np.clip(eta_out, -5.0, 5.0)), 0.0)
    return pt_out, eta_out, phi_out, energy_out, valid


def _apply_constituent_thresholds(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    energy: np.ndarray,
    mask: np.ndarray,
    *,
    cfg: CommonSmearCfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只施加常用 threshold，不施加 smear。"""
    pt_out = np.asarray(pt, dtype=np.float64).copy()
    eta_out = np.asarray(eta, dtype=np.float64).copy()
    phi_out = np.asarray(phi, dtype=np.float64).copy()
    energy_out = np.asarray(energy, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool).copy()

    valid &= pt_out >= float(cfg.pt_threshold_offline)
    valid &= pt_out >= float(cfg.pt_threshold_hlt)

    pt_out[~valid] = 0.0
    eta_out[~valid] = 0.0
    phi_out[~valid] = 0.0
    energy_out[~valid] = 0.0
    return pt_out, eta_out, phi_out, energy_out, valid


def build_smeared_jet_frame_feature_map(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
    cfg: Optional[CommonSmearCfg] = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """对前 N 个 jet 做 smear 后，再构造 jet-axis 坐标系下的特征。"""
    cfg = CommonSmearCfg() if cfg is None else cfg
    pt, eta, phi, energy, mask = _load_constituent_arrays(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
    )
    pt, eta, phi, energy, mask = _apply_constituent_smear(
        pt,
        eta,
        phi,
        energy,
        mask,
        cfg=cfg,
        seed=seed,
    )

    px = np.where(mask, pt * np.cos(phi), 0.0)
    py = np.where(mask, pt * np.sin(phi), 0.0)
    pz = np.where(mask, pt * np.sinh(eta), 0.0)
    jet_px = px.sum(axis=1, keepdims=True)
    jet_py = py.sum(axis=1, keepdims=True)
    jet_pz = pz.sum(axis=1, keepdims=True)
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    deta = eta - jet_eta
    dphi = wrap_dphi_np(phi - jet_phi)
    return {
        "dEta": deta[mask],
        "dPhi": dphi[mask],
    }


def build_jet_frame_feature_map(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
) -> dict[str, np.ndarray]:
    """仅用 constituent 的 pt/eta/phi/E 构造 jet-axis 坐标系下的特征。"""
    pt, eta, phi, energy, mask = _load_constituent_arrays(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
    )

    px = np.where(mask, pt * np.cos(phi), 0.0)
    py = np.where(mask, pt * np.sin(phi), 0.0)
    pz = np.where(mask, pt * np.sinh(eta), 0.0)

    jet_px = px.sum(axis=1, keepdims=True)
    jet_py = py.sum(axis=1, keepdims=True)
    jet_pz = pz.sum(axis=1, keepdims=True)
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)

    deta = eta - jet_eta
    dphi = wrap_dphi_np(phi - jet_phi)
    log_pt = np.log(np.clip(pt, 1e-12, None))
    log_e = np.log(np.clip(energy, 1e-12, None))

    return {
        "dEta": deta[mask],
        "dPhi": dphi[mask],
        "log_pt": log_pt[mask],
        "log_E": log_e[mask],
    }


def inspect_jet_frame_constituent_distributions(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
    bins: int = 120,
    logy: bool = False,
    percentiles: Sequence[float] = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9),
) -> dict[str, np.ndarray]:
    """检查 jet-axis 坐标系下 constituent 特征的分布。"""
    feature_map = build_jet_frame_feature_map(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    for ax, (name, values) in zip(axes, feature_map.items()):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        _print_distribution_stats(name, values, percentiles=percentiles)
        ax.hist(values, bins=int(bins), alpha=0.85)
        ax.set_title(f"{name} distribution in jet frame")
        ax.set_xlabel(name)
        ax.set_ylabel("counts")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()
    return feature_map


def inspect_standardized_jet_frame_constituent_distributions(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
    bins: int = 120,
    logy: bool = False,
    percentiles: Sequence[float] = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9),
) -> dict[str, dict[str, np.ndarray] | dict[str, float]]:
    """检查 jet-axis 后再做 mean/std 标准化的四维特征分布。"""
    feature_map = build_jet_frame_feature_map(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
    )

    standardized_map: dict[str, np.ndarray] = {}
    stats_map: dict[str, float] = {}
    for name, values in feature_map.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        mean = float(arr.mean())
        std = float(arr.std() + 1e-8)
        standardized = (arr - mean) / std
        standardized_map[name] = standardized
        stats_map[f"{name}_mean"] = mean
        stats_map[f"{name}_std"] = std

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    for ax, (name, values) in zip(axes, standardized_map.items()):
        _print_distribution_stats(f"{name}_std", values, percentiles=percentiles)
        ax.hist(values, bins=int(bins), alpha=0.85)
        ax.set_title(f"standardized {name} distribution in jet frame")
        ax.set_xlabel(f"standardized {name}")
        ax.set_ylabel("counts")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()
    return {
        "features": standardized_map,
        "stats": stats_map,
    }


def get_single_jet_constituents(
    h5_file: h5py.File,
    *,
    jet_idx: int,
    max_particles: int | None = None,
) -> dict[str, np.ndarray]:
    """读取单个 jet 的有效 constituent 四动量信息。"""
    pt_ds = h5_file["fjet_clus_pt"]
    s = int(min(int(pt_ds.shape[1]), max_particles)) if max_particles is not None else int(pt_ds.shape[1])

    pt = np.asarray(h5_file["fjet_clus_pt"][int(jet_idx), :s], dtype=np.float64)
    eta = np.asarray(h5_file["fjet_clus_eta"][int(jet_idx), :s], dtype=np.float64)
    phi = np.asarray(h5_file["fjet_clus_phi"][int(jet_idx), :s], dtype=np.float64)
    energy = np.asarray(h5_file["fjet_clus_E"][int(jet_idx), :s], dtype=np.float64)

    mask = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & np.isfinite(energy) & (pt > 0.0)
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    energy = energy[mask]
    momentum = pt * np.cosh(eta)

    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "E": energy,
        "p": momentum,
    }


def _single_jet_axis_features(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """基于单个 jet 的 constituent 计算 jet-axis 下的 dEta/dPhi。"""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    jet_px = px.sum()
    jet_py = py.sum()
    jet_pz = pz.sum()
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)
    deta = eta - jet_eta
    dphi = wrap_dphi_np(phi - jet_phi)
    return deta, dphi, float(jet_eta), float(jet_phi)


def get_jet_frame_standardization_stats(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
) -> dict[str, float]:
    """用前 N 个 jet 的 jet-frame dEta/dPhi 统计标准化参数。"""
    feature_map = build_jet_frame_feature_map(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
    )
    deta = np.asarray(feature_map["dEta"], dtype=np.float64)
    dphi = np.asarray(feature_map["dPhi"], dtype=np.float64)
    deta = deta[np.isfinite(deta)]
    dphi = dphi[np.isfinite(dphi)]
    return {
        "dEta_mean": float(deta.mean()),
        "dEta_std": float(deta.std() + 1e-8),
        "dPhi_mean": float(dphi.mean()),
        "dPhi_std": float(dphi.std() + 1e-8),
        "n_jets": int(n_jets),
        "max_particles": int(max_particles) if max_particles is not None else -1,
    }


def get_smeared_jet_frame_standardization_stats(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
    cfg: Optional[CommonSmearCfg] = None,
    seed: int = 42,
) -> dict[str, float]:
    """用 smear 后前 N 个 jet 的 jet-frame dEta/dPhi 统计标准化参数。"""
    feature_map = build_smeared_jet_frame_feature_map(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
        cfg=cfg,
        seed=seed,
    )
    deta = np.asarray(feature_map["dEta"], dtype=np.float64)
    dphi = np.asarray(feature_map["dPhi"], dtype=np.float64)
    deta = deta[np.isfinite(deta)]
    dphi = dphi[np.isfinite(dphi)]
    return {
        "dEta_mean": float(deta.mean()),
        "dEta_std": float(deta.std() + 1e-8),
        "dPhi_mean": float(dphi.mean()),
        "dPhi_std": float(dphi.std() + 1e-8),
        "n_jets": int(n_jets),
        "max_particles": int(max_particles) if max_particles is not None else -1,
    }


def get_thresholded_jet_frame_standardization_stats(
    h5_file: h5py.File,
    *,
    n_jets: int = 20000,
    max_particles: int | None = None,
    cfg: Optional[CommonSmearCfg] = None,
) -> dict[str, float]:
    """用 threshold 后前 N 个 jet 的 jet-frame dEta/dPhi 统计标准化参数。"""
    cfg = CommonSmearCfg() if cfg is None else cfg
    pt, eta, phi, energy, mask = _load_constituent_arrays(
        h5_file,
        n_jets=n_jets,
        max_particles=max_particles,
    )
    pt, eta, phi, energy, mask = _apply_constituent_thresholds(
        pt,
        eta,
        phi,
        energy,
        mask,
        cfg=cfg,
    )

    px = np.where(mask, pt * np.cos(phi), 0.0)
    py = np.where(mask, pt * np.sin(phi), 0.0)
    pz = np.where(mask, pt * np.sinh(eta), 0.0)
    jet_px = px.sum(axis=1, keepdims=True)
    jet_py = py.sum(axis=1, keepdims=True)
    jet_pz = pz.sum(axis=1, keepdims=True)
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8))
    jet_phi = np.arctan2(jet_py, jet_px)
    deta = eta - jet_eta
    dphi = wrap_dphi_np(phi - jet_phi)
    deta = np.asarray(deta[mask], dtype=np.float64)
    dphi = np.asarray(dphi[mask], dtype=np.float64)
    deta = deta[np.isfinite(deta)]
    dphi = dphi[np.isfinite(dphi)]
    return {
        "dEta_mean": float(deta.mean()),
        "dEta_std": float(deta.std() + 1e-8),
        "dPhi_mean": float(dphi.mean()),
        "dPhi_std": float(dphi.std() + 1e-8),
        "n_jets": int(n_jets),
        "max_particles": int(max_particles) if max_particles is not None else -1,
    }


def plot_single_jet_constituent_views(
    h5_file: h5py.File,
    *,
    jet_idx: Optional[int] = None,
    max_particles: int = 100,
    # rng_seed: int = 42,
    color_by: Sequence[str] = ("pt",),
) -> int:
    """画单个 jet 的 E-p 散点图和 jet-axis 平面图。"""
    if jet_idx is None:
        # rng = np.random.default_rng(int(rng_seed))
        jet_idx = np.random.randint(0, int(h5_file["fjet_clus_pt"].shape[0]))

    data = get_single_jet_constituents(
        h5_file,
        jet_idx=int(jet_idx),
        max_particles=max_particles,
    )
    pt = data["pt"]
    eta = data["eta"]
    phi = data["phi"]
    energy = data["E"]
    momentum = data["p"]
    const_idx = np.arange(len(momentum))

    print(f"Jet index: {jet_idx}")
    print(f"Constituents used: {len(momentum)}")
    if len(momentum) == 0:
        raise ValueError("No valid constituents found for this jet.")

    print(f"p min/max: {momentum.min():.6g} / {momentum.max():.6g}")
    print(f"E min/max: {energy.min():.6g} / {energy.max():.6g}")

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(momentum, energy, c=const_idx, cmap="viridis", s=28, alpha=0.85)
    lims = [0.0, max(float(np.max(momentum)), float(np.max(energy)))]
    plt.plot(lims, lims, "--", color="gray", lw=1.2, label="E = p")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|p| = pt * cosh(eta)")
    plt.ylabel("E")
    plt.title(f"Constituent E vs |p| in one jet (jet_idx={jet_idx})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    cbar = plt.colorbar(sc)
    cbar.set_label("Constituent index")
    plt.tight_layout()
    plt.show()

    deta, dphi, jet_eta, jet_phi = _single_jet_axis_features(pt, eta, phi)

    print(f"Jet axis eta/phi: {jet_eta:.6g} / {jet_phi:.6g}")

    color_map = {
        "pt": np.log10(np.clip(pt, 1e-12, None)),
        "|p|": np.log10(np.clip(momentum, 1e-12, None)),
        "E": np.log10(np.clip(energy, 1e-12, None)),
    }
    active = [str(name) for name in color_by if str(name) in color_map]
    if not active:
        raise ValueError(f"Unsupported color_by={list(color_by)}")

    fig, axes = plt.subplots(1, len(active), figsize=(6 * len(active), 4.8), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, active):
        values = color_map[name]
        sc = ax.scatter(deta, dphi, c=values, cmap="viridis", s=34, alpha=0.9)
        ax.set_title(f"dEta vs dPhi colored by log10({name})")
        ax.set_xlabel("dEta")
        ax.set_ylabel("dPhi")
        ax.grid(True, alpha=0.25)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f"log10({name})")

    fig.suptitle(f"Constituent coordinates in jet-axis frame (jet_idx={jet_idx})")
    plt.tight_layout()
    plt.show()
    return int(jet_idx)


def plot_single_jet_standardized_jet_frame(
    h5_file: h5py.File,
    *,
    jet_idx: Optional[int] = None,
    max_particles: int = 100,
    stats_n_jets: int = 200000,
    stats_max_particles: int | None = None,
    color_by: str = "pt",
) -> int:
    """画单个 jet 在标准化后的 jet-axis 坐标系中的 dEta/dPhi 散点图。"""
    if jet_idx is None:
        jet_idx = int(np.random.randint(0, int(h5_file["fjet_clus_pt"].shape[0])))

    data = get_single_jet_constituents(
        h5_file,
        jet_idx=int(jet_idx),
        max_particles=max_particles,
    )
    pt = data["pt"]
    eta = data["eta"]
    phi = data["phi"]
    energy = data["E"]
    momentum = data["p"]

    if len(pt) == 0:
        raise ValueError("No valid constituents found for this jet.")

    stats = get_jet_frame_standardization_stats(
        h5_file,
        n_jets=stats_n_jets,
        max_particles=max_particles if stats_max_particles is None else stats_max_particles,
    )
    deta, dphi, jet_eta, jet_phi = _single_jet_axis_features(pt, eta, phi)
    deta_std = (deta - float(stats["dEta_mean"])) / float(stats["dEta_std"])
    dphi_std = (dphi - float(stats["dPhi_mean"])) / float(stats["dPhi_std"])

    color_map = {
        "pt": np.log10(np.clip(pt, 1e-12, None)),
        "|p|": np.log10(np.clip(momentum, 1e-12, None)),
        "E": np.log10(np.clip(energy, 1e-12, None)),
    }
    color_key = str(color_by)
    if color_key not in color_map:
        raise ValueError(f"Unsupported color_by={color_by!r}")

    print(f"Jet index: {jet_idx}")
    print(f"Constituents used: {len(pt)}")
    print(f"Jet axis eta/phi: {jet_eta:.6g} / {jet_phi:.6g}")
    print(
        "Standardization stats: "
        f"dEta(mean={stats['dEta_mean']:.6g}, std={stats['dEta_std']:.6g}), "
        f"dPhi(mean={stats['dPhi_mean']:.6g}, std={stats['dPhi_std']:.6g})"
    )
    print(f"Stats jets used: {int(stats['n_jets'])}")

    plt.figure(figsize=(6.4, 5.4))
    sc = plt.scatter(
        deta_std,
        dphi_std,
        c=color_map[color_key],
        cmap="viridis",
        s=34,
        alpha=0.9,
    )
    plt.xlabel("standardized dEta")
    plt.ylabel("standardized dPhi")
    plt.title(
        "Constituent coordinates in standardized jet-axis frame "
        f"(jet_idx={jet_idx})"
    )
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label(f"log10({color_key})")
    plt.tight_layout()
    plt.show()
    return int(jet_idx)


def plot_single_jet_smeared_coordinate_views(
    h5_file: h5py.File,
    *,
    jet_idx: Optional[int] = None,
    max_particles: int = 100,
    stats_n_jets: int = 200000,
    stats_max_particles: int | None = None,
    color_by: str = "pt",
    cfg: Optional[CommonSmearCfg] = None,
    seed: int = 42,
) -> int:
    """画单个 jet 在 smear 后的 raw / jet-axis / standardized 三联图。"""
    cfg = CommonSmearCfg() if cfg is None else cfg
    if jet_idx is None:
        jet_idx = int(np.random.randint(0, int(h5_file["fjet_clus_pt"].shape[0])))

    pt_ds = h5_file["fjet_clus_pt"]
    s = int(min(int(pt_ds.shape[1]), max_particles))
    pt = np.asarray(h5_file["fjet_clus_pt"][int(jet_idx), :s], dtype=np.float64)[None, :]
    eta = np.asarray(h5_file["fjet_clus_eta"][int(jet_idx), :s], dtype=np.float64)[None, :]
    phi = np.asarray(h5_file["fjet_clus_phi"][int(jet_idx), :s], dtype=np.float64)[None, :]
    energy = np.asarray(h5_file["fjet_clus_E"][int(jet_idx), :s], dtype=np.float64)[None, :]
    mask = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & np.isfinite(energy) & (pt > 0.0)

    pt_sm, eta_sm, phi_sm, energy_sm, mask_sm = _apply_constituent_smear(
        pt,
        eta,
        phi,
        energy,
        mask,
        cfg=cfg,
        seed=seed + int(jet_idx),
    )

    pt_1d = pt_sm[0][mask_sm[0]]
    eta_1d = eta_sm[0][mask_sm[0]]
    phi_1d = phi_sm[0][mask_sm[0]]
    energy_1d = energy_sm[0][mask_sm[0]]
    momentum_1d = pt_1d * np.cosh(eta_1d)
    if len(pt_1d) == 0:
        raise ValueError("No valid smeared constituents found for this jet.")

    deta, dphi, jet_eta, jet_phi = _single_jet_axis_features(pt_1d, eta_1d, phi_1d)
    stats = get_smeared_jet_frame_standardization_stats(
        h5_file,
        n_jets=stats_n_jets,
        max_particles=max_particles if stats_max_particles is None else stats_max_particles,
        cfg=cfg,
        seed=seed,
    )
    deta_std = (deta - float(stats["dEta_mean"])) / float(stats["dEta_std"])
    dphi_std = (dphi - float(stats["dPhi_mean"])) / float(stats["dPhi_std"])

    color_map = {
        "pt": np.log10(np.clip(pt_1d, 1e-12, None)),
        "|p|": np.log10(np.clip(momentum_1d, 1e-12, None)),
        "E": np.log10(np.clip(energy_1d, 1e-12, None)),
    }
    color_key = str(color_by)
    if color_key not in color_map:
        raise ValueError(f"Unsupported color_by={color_by!r}")

    print(f"Jet index: {jet_idx}")
    print(f"Smeared constituents used: {len(pt_1d)}")
    print(
        "Smear cfg: "
        f"pt_thr_off={cfg.pt_threshold_offline}, pt_thr_hlt={cfg.pt_threshold_hlt}, "
        f"pt_res={cfg.pt_resolution}, eta_res={cfg.eta_resolution}, phi_res={cfg.phi_resolution}"
    )
    print(f"Jet axis eta/phi after smear: {jet_eta:.6g} / {jet_phi:.6g}")
    print(
        "Standardization stats after smear: "
        f"dEta(mean={stats['dEta_mean']:.6g}, std={stats['dEta_std']:.6g}), "
        f"dPhi(mean={stats['dPhi_mean']:.6g}, std={stats['dPhi_std']:.6g})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    plot_specs = [
        ("raw", eta_1d, phi_1d, "eta", "phi"),
        ("jet-axis", deta, dphi, "dEta", "dPhi"),
        ("std", deta_std, dphi_std, "standardized dEta", "standardized dPhi"),
    ]
    values = color_map[color_key]

    for ax, (title, x, y, xlabel, ylabel) in zip(axes, plot_specs):
        hb = ax.hexbin(
            x,
            y,
            C=values,
            gridsize=35,
            reduce_C_function=np.mean,
            cmap="viridis",
            mincnt=1,
        )
        ax.set_title(f"{title} coordinates after smear")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label(f"mean log10({color_key})")

    fig.suptitle(f"Smeared constituent views (jet_idx={jet_idx})")
    plt.tight_layout()
    plt.show()
    return int(jet_idx)


def plot_single_jet_smeared_std_comparison(
    h5_file: h5py.File,
    *,
    jet_idx: Optional[int] = None,
    max_particles: int = 100,
    stats_n_jets: int = 200000,
    stats_max_particles: int | None = None,
    cfg: Optional[CommonSmearCfg] = None,
    seed: int = 42,
    cmap: str = "turbo",
) -> int:
    """在同一张 std 空间图中展示 smear 前后的对应位移。"""
    cfg = CommonSmearCfg() if cfg is None else cfg
    if jet_idx is None:
        jet_idx = int(np.random.randint(0, int(h5_file["fjet_clus_pt"].shape[0])))

    pt_ds = h5_file["fjet_clus_pt"]
    s = int(min(int(pt_ds.shape[1]), max_particles))
    pt = np.asarray(h5_file["fjet_clus_pt"][int(jet_idx), :s], dtype=np.float64)[None, :]
    eta = np.asarray(h5_file["fjet_clus_eta"][int(jet_idx), :s], dtype=np.float64)[None, :]
    phi = np.asarray(h5_file["fjet_clus_phi"][int(jet_idx), :s], dtype=np.float64)[None, :]
    energy = np.asarray(h5_file["fjet_clus_E"][int(jet_idx), :s], dtype=np.float64)[None, :]
    mask = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi) & np.isfinite(energy) & (pt > 0.0)

    pt_base, eta_base, phi_base, energy_base, mask_base = _apply_constituent_thresholds(
        pt,
        eta,
        phi,
        energy,
        mask,
        cfg=cfg,
    )
    pt_sm, eta_sm, phi_sm, energy_sm, mask_sm = _apply_constituent_smear(
        pt,
        eta,
        phi,
        energy,
        mask,
        cfg=cfg,
        seed=seed + int(jet_idx),
    )

    shared = mask_base[0] & mask_sm[0]
    pt_base_1d = pt_base[0][shared]
    eta_base_1d = eta_base[0][shared]
    phi_base_1d = phi_base[0][shared]
    pt_sm_1d = pt_sm[0][shared]
    eta_sm_1d = eta_sm[0][shared]
    phi_sm_1d = phi_sm[0][shared]

    if len(pt_base_1d) == 0:
        raise ValueError("No valid constituents found for this jet after thresholding.")

    stats = get_thresholded_jet_frame_standardization_stats(
        h5_file,
        n_jets=stats_n_jets,
        max_particles=max_particles if stats_max_particles is None else stats_max_particles,
        cfg=cfg,
    )

    deta_base, dphi_base, jet_eta_base, jet_phi_base = _single_jet_axis_features(
        pt_base_1d,
        eta_base_1d,
        phi_base_1d,
    )
    deta_sm, dphi_sm, jet_eta_sm, jet_phi_sm = _single_jet_axis_features(
        pt_sm_1d,
        eta_sm_1d,
        phi_sm_1d,
    )

    deta_base_std = (deta_base - float(stats["dEta_mean"])) / float(stats["dEta_std"])
    dphi_base_std = (dphi_base - float(stats["dPhi_mean"])) / float(stats["dPhi_std"])
    deta_sm_std = (deta_sm - float(stats["dEta_mean"])) / float(stats["dEta_std"])
    dphi_sm_std = (dphi_sm - float(stats["dPhi_mean"])) / float(stats["dPhi_std"])

    delta_std = np.sqrt((deta_sm_std - deta_base_std) ** 2 + (dphi_sm_std - dphi_base_std) ** 2)
    segments = np.stack(
        [
            np.column_stack([deta_base_std, dphi_base_std]),
            np.column_stack([deta_sm_std, dphi_sm_std]),
        ],
        axis=1,
    )
    norm = Normalize(vmin=float(np.min(delta_std)), vmax=float(np.max(delta_std) + 1e-12))

    print(f"Jet index: {jet_idx}")
    print(f"Matched constituents used: {len(pt_base_1d)}")
    print(
        "Smear cfg: "
        f"pt_thr_off={cfg.pt_threshold_offline}, pt_thr_hlt={cfg.pt_threshold_hlt}, "
        f"pt_res={cfg.pt_resolution}, eta_res={cfg.eta_resolution}, phi_res={cfg.phi_resolution}"
    )
    print(f"Base jet axis eta/phi: {jet_eta_base:.6g} / {jet_phi_base:.6g}")
    print(f"Smeared jet axis eta/phi: {jet_eta_sm:.6g} / {jet_phi_sm:.6g}")
    print(f"Standardized shift magnitude: mean={delta_std.mean():.6g}, max={delta_std.max():.6g}")

    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(
        deta_base_std,
        dphi_base_std,
        s=20,
        c="#d9d9d9",
        alpha=0.7,
        label="before smear",
        zorder=1,
    )

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=1.6,
        alpha=0.85,
        zorder=2,
    )
    lc.set_array(delta_std)
    ax = plt.gca()
    ax.add_collection(lc)

    sc = plt.scatter(
        deta_sm_std,
        dphi_sm_std,
        c=delta_std,
        cmap=cmap,
        norm=norm,
        s=34,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.3,
        label="after smear",
        zorder=3,
    )
    plt.xlabel("standardized dEta")
    plt.ylabel("standardized dPhi")
    plt.title(f"Smear shift in standardized jet frame (jet_idx={jet_idx})")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    cbar = plt.colorbar(sc)
    cbar.set_label("shift magnitude in std space")
    plt.tight_layout()
    plt.show()
    return int(jet_idx)
