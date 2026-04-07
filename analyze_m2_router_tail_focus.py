#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

from analyze_m2_router_signal_sweep import (
    _clip_probs,
    _load_ckpt_state,
    _infer_dual_input_dims,
    _build_train_file_list,
    _offline_mask,
    _parse_corruptions,
    _apply_corruption_batch,
    _init_shift_acc,
    _update_shift_acc,
    _shift_features_from_acc,
    _build_raw_features,
    _jet_level_features,
    _stack_features,
    _threshold_at_target_tpr,
    _lowfpr_cost,
    _score_metrics,
    _save_csv,
    _infer_scores_and_diag,
)


def _deepcopy_cfg() -> Dict:
    return json.loads(json.dumps(BASE_CONFIG))


def _choose_best_threshold_tail(
    y_cal: np.ndarray,
    p_h_cal: np.ndarray,
    p_j_cal: np.ndarray,
    q_cal: np.ndarray,
    tail_mask_cal: np.ndarray,
    fallback: str,
    objective: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    q_tail = q_cal[tail_mask_cal]
    if q_tail.size == 0:
        raise RuntimeError("No tail jets in calibration split for threshold selection")

    thr_grid = np.unique(
        np.concatenate(
            [
                np.array([0.0], dtype=np.float64),
                np.quantile(q_tail, np.linspace(0.01, 0.99, 199)).astype(np.float64),
                np.array([1.0], dtype=np.float64),
            ]
        )
    )

    if str(fallback) == "hlt":
        base_cal = p_h_cal.copy()
    else:
        base_cal = p_j_cal.copy()

    rows: List[Dict[str, object]] = []
    best = None

    for d in [">=", "<="]:
        for t in thr_grid:
            if d == ">=":
                route_joint = q_cal >= float(t)
            else:
                route_joint = q_cal <= float(t)

            score_cal = base_cal.copy()
            use = tail_mask_cal
            score_cal[use] = np.where(route_joint[use], p_j_cal[use], p_h_cal[use])

            m = _score_metrics(y_cal, score_cal, p_h_cal)
            row = {
                "direction": d,
                "threshold": float(t),
                "route_frac_tail_cal": float(np.mean(route_joint[use])) if np.any(use) else float("nan"),
                "tail_coverage_cal": float(np.mean(use)),
                "auc_cal": float(m["auc"]),
                "fpr30_cal": float(m["fpr30"]),
                "fpr50_cal": float(m["fpr50"]),
            }
            rows.append(row)

            if best is None:
                best = row
            else:
                if objective == "auc":
                    key = (row["auc_cal"], -row["fpr50_cal"], -row["fpr30_cal"])
                    key_b = (best["auc_cal"], -best["fpr50_cal"], -best["fpr30_cal"])
                    if key > key_b:
                        best = row
                else:
                    key = (-row["fpr50_cal"], row["auc_cal"], -row["fpr30_cal"])
                    key_b = (-best["fpr50_cal"], best["auc_cal"], -best["fpr30_cal"])
                    if key > key_b:
                        best = row

    assert best is not None
    rows_sorted = sorted(rows, key=lambda r: (r["fpr50_cal"], -r["auc_cal"], r["fpr30_cal"]))
    return best, rows_sorted


def _apply_tail_routing(
    p_h: np.ndarray,
    p_j: np.ndarray,
    q: np.ndarray,
    tail_mask: np.ndarray,
    direction: str,
    threshold: float,
    fallback: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if direction == ">=":
        route_joint = q >= float(threshold)
    elif direction == "<=":
        route_joint = q <= float(threshold)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    if str(fallback) == "hlt":
        score = p_h.copy()
    else:
        score = p_j.copy()

    use = np.asarray(tail_mask, dtype=bool)
    score[use] = np.where(route_joint[use], p_j[use], p_h[use])
    return score, route_joint


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class RouterLinear(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(1)


def _predict_q(model: nn.Module, x_np: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = np.zeros((x_np.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, x_np.shape[0], int(batch_size)):
            e = min(x_np.shape[0], s + int(batch_size))
            xb = torch.from_numpy(x_np[s:e]).to(device=device, dtype=torch.float32)
            logits = model(xb)
            out[s:e] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tail-focused router train/eval (train+route only in HLT/Joint TPR-tail mask)")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--train_path", type=str, default="./data")

    ap.add_argument("--router_offset_jets", type=int, default=375000)
    ap.add_argument("--router_n_analysis", type=int, default=300000)
    ap.add_argument("--router_n_test", type=int, default=200000)
    ap.add_argument("--ref_offset_jets", type=int, default=0)
    ap.add_argument("--ref_n_jets", type=int, default=375000)
    ap.add_argument("--max_constits", type=int, default=100)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hlt_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--corrected_weight_floor", type=float, default=1e-4)

    ap.add_argument(
        "--corruptions",
        type=str,
        default="pt_noise:0.03,pt_noise:0.06,eta_phi_jitter:0.02,eta_phi_jitter:0.05,dropout:0.05,dropout:0.10,merge:0.10,merge:0.20,global_scale:0.03",
    )
    ap.add_argument("--use_corruption_features", action="store_true")

    ap.add_argument("--router_cal_frac", type=float, default=0.2)
    ap.add_argument("--cost_alpha_neg", type=float, default=4.0)
    ap.add_argument("--cost_tau", type=float, default=0.02)

    ap.add_argument("--tail_tpr_cut", type=float, default=0.6)
    ap.add_argument("--fallback_non_tail", type=str, default="joint", choices=["joint", "hlt"])
    ap.add_argument("--selection_metric", type=str, default="fpr50", choices=["fpr50", "auc"])
    ap.add_argument("--router_model", type=str, default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--router_hidden", type=int, default=256)
    ap.add_argument("--router_dropout", type=float, default=0.10)
    ap.add_argument("--router_epochs", type=int, default=40)
    ap.add_argument("--router_patience", type=int, default=8)
    ap.add_argument("--router_lr", type=float, default=1e-3)
    ap.add_argument("--router_weight_decay", type=float, default=1e-4)
    ap.add_argument("--router_batch_size", type=int, default=4096)

    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    ap.add_argument("--save_per_jet_npz", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (run_dir / "router_tail_focus_300k200k")
    )
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

    # ---------------- Standardization stats ----------------
    ref_need = int(args.ref_offset_jets + args.ref_n_jets)
    const_ref_raw, labels_ref_all = load_raw_constituents_from_h5(
        train_files,
        max_jets=ref_need,
        max_constits=int(args.max_constits),
    )
    const_ref_raw = const_ref_raw[int(args.ref_offset_jets) : int(args.ref_offset_jets) + int(args.ref_n_jets)]
    labels_ref = labels_ref_all[int(args.ref_offset_jets) : int(args.ref_offset_jets) + int(args.ref_n_jets)].astype(np.int64)

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

    # ---------------- Router dataset ----------------
    n_total_router = int(args.router_n_analysis + args.router_n_test)
    need = int(args.router_offset_jets + n_total_router)
    const_raw_all, labels_all = load_raw_constituents_from_h5(
        train_files,
        max_jets=need,
        max_constits=int(args.max_constits),
    )
    const_raw = const_raw_all[int(args.router_offset_jets) : int(args.router_offset_jets) + n_total_router]
    labels = labels_all[int(args.router_offset_jets) : int(args.router_offset_jets) + n_total_router].astype(np.int64)

    const_off, mask_off = _offline_mask(const_raw, float(cfg["hlt_effects"]["pt_threshold_offline"]))
    hlt_const, hlt_mask, _hlt_stats_gen, _ = apply_hlt_effects_realistic_nomap(
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
        raise FileNotFoundError("Missing checkpoints in run_dir. Need baseline.pt, offline_reconstructor.pt, dual_joint.pt")

    baseline_sd = _load_ckpt_state(baseline_ckpt, device)
    reco_sd = _load_ckpt_state(reco_ckpt, device)
    dual_sd = _load_ckpt_state(dual_ckpt, device)

    dual_in_a, dual_in_b = _infer_dual_input_dims(dual_sd)
    if int(dual_in_a) != 7:
        raise RuntimeError(f"Expected dual input_dim_a=7, got {dual_in_a}")
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

    # ---------------- Optional corruption features ----------------
    shift_feats: Dict[str, np.ndarray] = {}
    corruption_list: List[Tuple[str, float]] = []
    if bool(args.use_corruption_features):
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

    # ---------------- Assemble features ----------------
    feature_dict = _build_raw_features(clean.p_hlt, clean.p_joint, clean.diag)
    feature_dict.update(_jet_level_features(hlt_const, hlt_mask, feat_hlt_std))
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

    ph_fit = ph_a[idx_fit]
    pj_fit = pj_a[idx_fit]
    ph_cal = ph_a[idx_cal]
    pj_cal = pj_a[idx_cal]

    X_fit = X_a[idx_fit]
    X_cal = X_a[idx_cal]

    # Low-FPR routing target from fit split.
    thr_h30 = _threshold_at_target_tpr(y_fit, ph_fit, 0.30)
    thr_h50 = _threshold_at_target_tpr(y_fit, ph_fit, 0.50)
    thr_j30 = _threshold_at_target_tpr(y_fit, pj_fit, 0.30)
    thr_j50 = _threshold_at_target_tpr(y_fit, pj_fit, 0.50)

    c_h_fit = _lowfpr_cost(y_fit, ph_fit, thr_h30, thr_h50, args.cost_alpha_neg, args.cost_tau)
    c_j_fit = _lowfpr_cost(y_fit, pj_fit, thr_j30, thr_j50, args.cost_alpha_neg, args.cost_tau)
    z_fit = (c_j_fit < c_h_fit).astype(np.int64)

    # Tail mask definition: HLT tail OR Joint tail using TPR cut.
    tail_tpr = float(args.tail_tpr_cut)
    thr_h_tail = _threshold_at_target_tpr(y_fit, ph_fit, tail_tpr)
    thr_j_tail = _threshold_at_target_tpr(y_fit, pj_fit, tail_tpr)

    tail_fit = (ph_fit >= thr_h_tail) | (pj_fit >= thr_j_tail)
    tail_cal = (ph_cal >= thr_h_tail) | (pj_cal >= thr_j_tail)
    tail_test = (ph_t >= thr_h_tail) | (pj_t >= thr_j_tail)

    if int(np.sum(tail_fit)) < 2000:
        raise RuntimeError(
            f"Too few tail training jets ({int(np.sum(tail_fit))}). Consider lower tail cut or more data."
        )

    # Train router on tail-only jets with epoch-based optimization.
    scaler = StandardScaler()
    scaler.fit(X_fit[tail_fit])
    X_fit_s = scaler.transform(X_fit).astype(np.float32)
    X_cal_s = scaler.transform(X_cal).astype(np.float32)
    X_t_s = scaler.transform(X_t).astype(np.float32)

    x_train_tail = X_fit_s[tail_fit]
    y_train_tail = z_fit[tail_fit].astype(np.float32)
    n_pos = int(np.sum(y_train_tail > 0.5))
    n_neg = int(y_train_tail.shape[0] - n_pos)
    pos_weight = float(n_neg / max(n_pos, 1))

    if str(args.router_model) == "linear":
        router = RouterLinear(in_dim=int(X_fit_s.shape[1])).to(device)
    else:
        router = RouterMLP(
            in_dim=int(X_fit_s.shape[1]),
            hidden=int(args.router_hidden),
            dropout=float(args.router_dropout),
        ).to(device)

    opt = torch.optim.AdamW(
        router.parameters(),
        lr=float(args.router_lr),
        weight_decay=float(args.router_weight_decay),
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))

    idx_perm = np.arange(x_train_tail.shape[0])
    best_epoch = -1
    best_state = None
    best_sel = float("-inf") if str(args.selection_metric) == "auc" else float("inf")
    best_q_cal = None
    best_q_test = None
    best_threshold_row = None
    history: List[Dict[str, float]] = []
    bad_epochs = 0

    for ep in range(1, int(args.router_epochs) + 1):
        np.random.shuffle(idx_perm)
        router.train()
        train_loss_sum = 0.0
        train_seen = 0

        for s in range(0, idx_perm.shape[0], int(args.router_batch_size)):
            e = min(idx_perm.shape[0], s + int(args.router_batch_size))
            bi = idx_perm[s:e]
            xb = torch.from_numpy(x_train_tail[bi]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_train_tail[bi]).to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            logits = router(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            bsz = int(xb.shape[0])
            train_loss_sum += float(loss.detach().cpu().item()) * bsz
            train_seen += bsz

        train_loss = float(train_loss_sum / max(train_seen, 1))
        q_cal_ep = _predict_q(router, X_cal_s, device, batch_size=max(4096, int(args.router_batch_size)))
        q_test_ep = _predict_q(router, X_t_s, device, batch_size=max(4096, int(args.router_batch_size)))

        best_ep_row, _ = _choose_best_threshold_tail(
            y_cal=y_cal,
            p_h_cal=ph_cal,
            p_j_cal=pj_cal,
            q_cal=q_cal_ep,
            tail_mask_cal=tail_cal,
            fallback=str(args.fallback_non_tail),
            objective=str(args.selection_metric),
        )
        score_cal_ep, _ = _apply_tail_routing(
            p_h=ph_cal,
            p_j=pj_cal,
            q=q_cal_ep,
            tail_mask=tail_cal,
            direction=str(best_ep_row["direction"]),
            threshold=float(best_ep_row["threshold"]),
            fallback=str(args.fallback_non_tail),
        )
        m_cal_ep = _score_metrics(y_cal, score_cal_ep, ph_cal)
        sel = float(m_cal_ep["auc"]) if str(args.selection_metric) == "auc" else float(m_cal_ep["fpr50"])
        improved = (sel > best_sel) if str(args.selection_metric) == "auc" else (sel < best_sel)
        if improved:
            best_sel = sel
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
            best_q_cal = q_cal_ep.copy()
            best_q_test = q_test_ep.copy()
            best_threshold_row = dict(best_ep_row)
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": int(ep),
                "train_loss": float(train_loss),
                "cal_auc": float(m_cal_ep["auc"]),
                "cal_fpr30": float(m_cal_ep["fpr30"]),
                "cal_fpr50": float(m_cal_ep["fpr50"]),
                "cal_tail_route_frac": float(best_ep_row.get("route_frac_tail_cal", float("nan"))),
            }
        )

        print(
            f"Router ep{ep:02d}: train_loss={train_loss:.5f} | "
            f"cal_auc={m_cal_ep['auc']:.6f}, cal_fpr50={m_cal_ep['fpr50']:.6f}, "
            f"tail_route_frac={best_ep_row.get('route_frac_tail_cal', float('nan')):.4f}"
        )

        if bad_epochs >= int(args.router_patience):
            print(f"Early stopping router at epoch {ep} (patience={args.router_patience})")
            break

    if best_state is None or best_q_cal is None or best_q_test is None or best_threshold_row is None:
        raise RuntimeError("Router training failed to produce a valid best checkpoint")

    router.load_state_dict(best_state, strict=True)
    q_cal = best_q_cal
    q_test = best_q_test

    best, sweep_rows = _choose_best_threshold_tail(
        y_cal=y_cal,
        p_h_cal=ph_cal,
        p_j_cal=pj_cal,
        q_cal=q_cal,
        tail_mask_cal=tail_cal,
        fallback=str(args.fallback_non_tail),
        objective=str(args.selection_metric),
    )
    # Keep selected threshold consistent with best epoch checkpoint when available.
    best = dict(best_threshold_row)

    # Selected route metrics.
    score_cal_sel, route_joint_cal = _apply_tail_routing(
        p_h=ph_cal,
        p_j=pj_cal,
        q=q_cal,
        tail_mask=tail_cal,
        direction=str(best["direction"]),
        threshold=float(best["threshold"]),
        fallback=str(args.fallback_non_tail),
    )
    score_test_sel, route_joint_test = _apply_tail_routing(
        p_h=ph_t,
        p_j=pj_t,
        q=q_test,
        tail_mask=tail_test,
        direction=str(best["direction"]),
        threshold=float(best["threshold"]),
        fallback=str(args.fallback_non_tail),
    )

    m_sel_cal = _score_metrics(y_cal, score_cal_sel, ph_cal)
    m_sel_test = _score_metrics(y_t, score_test_sel, ph_t)

    # Baselines/refs on test.
    if str(args.fallback_non_tail) == "hlt":
        p_base_test = ph_t
    else:
        p_base_test = pj_t

    # hard@0.5 tail-only
    score_test_h05, _ = _apply_tail_routing(
        p_h=ph_t,
        p_j=pj_t,
        q=q_test,
        tail_mask=tail_test,
        direction=">=",
        threshold=0.5,
        fallback=str(args.fallback_non_tail),
    )

    # softmix tail-only
    score_test_soft_tail = p_base_test.copy()
    score_test_soft_tail[tail_test] = q_test[tail_test] * pj_t[tail_test] + (1.0 - q_test[tail_test]) * ph_t[tail_test]

    metrics_test = {
        "hlt": _score_metrics(y_t, ph_t, ph_t),
        "joint": _score_metrics(y_t, pj_t, ph_t),
        "tail_hard_selected": m_sel_test,
        "tail_hard_05": _score_metrics(y_t, score_test_h05, ph_t),
        "tail_softmix": _score_metrics(y_t, score_test_soft_tail, ph_t),
    }

    # save sweep top rows
    _save_csv(out_dir / "tail_router_threshold_sweep_cal.csv", sweep_rows[:300])

    report = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "settings": vars(args),
        "n_total_router": int(n_total_router),
        "split": {
            "analysis": int(len(idx_analysis)),
            "test": int(len(idx_test)),
            "fit": int(len(idx_fit)),
            "cal": int(len(idx_cal)),
        },
        "thresholds": {
            "hlt_tpr30": float(thr_h30),
            "hlt_tpr50": float(thr_h50),
            "joint_tpr30": float(thr_j30),
            "joint_tpr50": float(thr_j50),
            "hlt_tpr_tail": float(thr_h_tail),
            "joint_tpr_tail": float(thr_j_tail),
        },
        "tail_mask_rates": {
            "fit": float(np.mean(tail_fit)),
            "cal": float(np.mean(tail_cal)),
            "test": float(np.mean(tail_test)),
            "n_fit_tail": int(np.sum(tail_fit)),
            "n_cal_tail": int(np.sum(tail_cal)),
            "n_test_tail": int(np.sum(tail_test)),
        },
        "selected_threshold": best,
        "router_training": {
            "model": str(args.router_model),
            "best_epoch": int(best_epoch),
            "best_selection_value": float(best_sel),
            "n_train_tail": int(np.sum(tail_fit)),
            "pos_weight": float(pos_weight),
            "history": history,
        },
        "selected_metrics": {
            "cal": m_sel_cal,
            "test": m_sel_test,
            "joint_route_frac_cal_tail": float(np.mean(route_joint_cal[tail_cal])) if np.any(tail_cal) else float("nan"),
            "joint_route_frac_test_tail": float(np.mean(route_joint_test[tail_test])) if np.any(tail_test) else float("nan"),
        },
        "metrics_test": metrics_test,
        "files": {
            "threshold_sweep_cal_csv": str((out_dir / "tail_router_threshold_sweep_cal.csv").resolve()),
        },
    }

    rep_path = (
        Path(args.report_json).expanduser().resolve()
        if str(args.report_json).strip()
        else (out_dir / "tail_router_report.json")
    )
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with rep_path.open("w") as f:
        json.dump(report, f, indent=2)

    if bool(args.save_per_jet_npz):
        np.savez_compressed(
            out_dir / "tail_router_perjet.npz",
            labels=labels.astype(np.int8),
            idx_analysis=idx_analysis.astype(np.int64),
            idx_test=idx_test.astype(np.int64),
            preds_hlt=clean.p_hlt.astype(np.float32),
            preds_joint=clean.p_joint.astype(np.float32),
            tail_mask_test=tail_test.astype(np.int8),
            q_test=q_test.astype(np.float32),
            selected_score_test=score_test_sel.astype(np.float32),
        )

    print("=" * 72)
    print("Tail-Focused Router Analysis")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Out dir: {out_dir}")
    print(f"N analysis/test: {len(idx_analysis)} / {len(idx_test)}")
    print(
        f"Tail cut TPR<{tail_tpr:.3f} | tail rates fit/cal/test: "
        f"{np.mean(tail_fit):.3f}/{np.mean(tail_cal):.3f}/{np.mean(tail_test):.3f}"
    )
    print(
        f"Selected threshold: dir={best['direction']} thr={best['threshold']:.6f} | "
        f"cal FPR50={m_sel_cal['fpr50']:.6f} AUC={m_sel_cal['auc']:.6f}"
    )
    print()
    print("Held-out test:")
    for k in ["hlt", "joint", "tail_hard_selected", "tail_hard_05", "tail_softmix"]:
        m = metrics_test[k]
        print(f"  {k:18s} AUC={m['auc']:.6f} FPR30={m['fpr30']:.6f} FPR50={m['fpr50']:.6f}")
    print()
    print(f"Saved report: {rep_path}")
    print(f"Saved sweep CSV: {out_dir / 'tail_router_threshold_sweep_cal.csv'}")
    if bool(args.save_per_jet_npz):
        print(f"Saved per-jet NPZ: {out_dir / 'tail_router_perjet.npz'}")


if __name__ == "__main__":
    main()
