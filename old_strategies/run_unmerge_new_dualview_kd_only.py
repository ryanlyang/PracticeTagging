import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import utils
from unmerge_new_ideas import (
    CONFIG,
    RANDOM_SEED,
    ETA_IDX,
    PHI_IDX,
    PT_IDX,
    apply_hlt_effects_with_tracking,
    compute_features,
    get_stats,
    standardize,
    ParticleTransformer,
    MergeCountPredictor,
    UnmergePredictor,
    DualViewCrossAttnClassifier,
    DualViewKDDataset,
    UnmergeKDDataset,
    JetDataset,
    build_unmerged_dataset,
    predict_counts,
    get_scheduler,
    train_classifier,
    eval_classifier,
    train_kd_epoch,
    evaluate_kd,
    evaluate_bce_loss,
    evaluate_bce_loss_unmerged,
    train_kd_epoch_dual,
    evaluate_kd_dual,
    evaluate_bce_loss_dual,
    self_train_student_dual,
    self_train_student,
    kd_loss_conf_weighted,
)


class KDProjector(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hid = max(out_dim, in_dim // 2)
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hid),
            torch.nn.GELU(),
            torch.nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _extract_dual_outputs(model, feat_a, mask_a, feat_b, mask_b):
    bsz, seq_len, _ = feat_a.shape
    h_a = model.input_proj_a(feat_a.view(-1, feat_a.size(-1))).view(bsz, seq_len, -1)
    h_b = model.input_proj_b(feat_b.view(-1, feat_b.size(-1))).view(bsz, seq_len, -1)
    h_a = model.encoder_a(h_a, src_key_padding_mask=~mask_a)
    h_b = model.encoder_b(h_b, src_key_padding_mask=~mask_b)
    query = model.pool_query.expand(bsz, -1, -1)
    pooled_a, _ = model.pool_attn_a(query, h_a, h_a, key_padding_mask=~mask_a, need_weights=False)
    pooled_b, _ = model.pool_attn_b(query, h_b, h_b, key_padding_mask=~mask_b, need_weights=False)
    cross_a, _ = model.cross_a_to_b(pooled_a, h_b, h_b, key_padding_mask=~mask_b, need_weights=False)
    cross_b, _ = model.cross_b_to_a(pooled_b, h_a, h_a, key_padding_mask=~mask_a, need_weights=False)
    fused = torch.cat([pooled_a, pooled_b, cross_a, cross_b], dim=-1).squeeze(1)
    fused = model.norm(fused)
    logits = model.classifier(fused).squeeze(1)
    return {
        "logits": logits,
        "pooled_a": pooled_a.squeeze(1),
        "pooled_b": pooled_b.squeeze(1),
        "cross_a": cross_a.squeeze(1),
        "cross_b": cross_b.squeeze(1),
        "fused": fused,
    }


def _pairwise_cosine_sim(z):
    z = torch.nn.functional.normalize(z, p=2, dim=-1)
    return z @ z.transpose(0, 1)


def _train_kd_epoch_dual_advanced(student, teacher, loader, opt, device, kd_cfg, proj_fused=None):
    student.train()
    teacher.eval()
    if proj_fused is not None:
        proj_fused.train()

    total_loss = 0.0
    preds, labs = [], []

    T = kd_cfg["temperature"]
    a_kd = kd_cfg["alpha_kd"]
    w_feat = kd_cfg.get("w_feat", 0.0)
    w_rel = kd_cfg.get("w_rel", 0.0)
    w_branch = kd_cfg.get("w_branch", 0.0)

    for batch in loader:
        xa = batch["feat_a"].to(device)
        ma = batch["mask_a"].to(device)
        xb = batch["feat_b"].to(device)
        mb = batch["mask_b"].to(device)
        x_o = batch["off"].to(device)
        m_o = batch["mask_off"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            t_logits = teacher(x_o, m_o).squeeze(1)
            _, t_emb_off = teacher(x_o, m_o, return_embedding=True)
            t_emb_a = t_emb_b = None
            if w_branch > 0.0:
                _, t_emb_a = teacher(xa[..., :teacher.input_dim], ma, return_embedding=True)
                _, t_emb_b = teacher(xb[..., :teacher.input_dim], mb, return_embedding=True)

        opt.zero_grad()
        out = _extract_dual_outputs(student, xa, ma, xb, mb)
        s_logits = out["logits"]

        loss_hard = torch.nn.functional.binary_cross_entropy_with_logits(s_logits, y)
        if kd_cfg["conf_weighted"]:
            loss_kd = kd_loss_conf_weighted(s_logits, t_logits, T)
        else:
            s_soft = torch.sigmoid(s_logits / T)
            t_soft = torch.sigmoid(t_logits / T)
            loss_kd = torch.nn.functional.binary_cross_entropy(s_soft, t_soft) * (T ** 2)

        loss = (1.0 - a_kd) * loss_hard + a_kd * loss_kd

        if (w_feat > 0.0 or w_rel > 0.0) and proj_fused is not None:
            s_emb = proj_fused(out["fused"])
            if w_feat > 0.0:
                loss_feat = torch.nn.functional.smooth_l1_loss(s_emb, t_emb_off)
                loss = loss + w_feat * loss_feat
            if w_rel > 0.0 and s_emb.size(0) > 1:
                s_sim = _pairwise_cosine_sim(s_emb)
                t_sim = _pairwise_cosine_sim(t_emb_off)
                loss_rel = torch.nn.functional.mse_loss(s_sim, t_sim)
                loss = loss + w_rel * loss_rel

        if w_branch > 0.0 and t_emb_a is not None and t_emb_b is not None:
            loss_branch = (
                torch.nn.functional.smooth_l1_loss(out["pooled_a"], t_emb_a)
                + torch.nn.functional.smooth_l1_loss(out["pooled_b"], t_emb_b)
                + 0.5 * torch.nn.functional.smooth_l1_loss(out["cross_a"], t_emb_b)
                + 0.5 * torch.nn.functional.smooth_l1_loss(out["cross_b"], t_emb_a)
            )
            loss = loss + w_branch * loss_branch

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * len(y)
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    preds = np.array(preds)
    labs = np.array(labs)
    auc = 0.0
    if len(np.unique(labs)) > 1:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labs, preds)
    return total_loss / max(1, len(preds)), auc


def _build_kd_sweep_configs(base_kd_cfg, max_runs=30):
    cfgs = []
    # 1) Explicit default (must be first)
    cfgs.append({"name": "kd_default", "w_feat": 0.0, "w_rel": 0.0, "w_branch": 0.0})

    # core single methods and combinations
    core = [
        ("feat_only", 0.10, 0.0, 0.0),
        ("rel_only", 0.0, 0.10, 0.0),
        ("branch_only", 0.0, 0.0, 0.10),
        ("feat_rel", 0.10, 0.10, 0.0),
        ("feat_branch", 0.10, 0.0, 0.10),
        ("rel_branch", 0.0, 0.10, 0.10),
        ("feat_rel_branch", 0.10, 0.10, 0.10),
    ]
    for n, wf, wr, wb in core:
        cfgs.append({"name": n, "w_feat": wf, "w_rel": wr, "w_branch": wb})

    # temperature/alpha variants
    variants = [
        {"temperature": 2.0},
        {"temperature": 4.0},
        {"alpha_kd": 0.3},
        {"alpha_kd": 0.7},
        {"conf_weighted": False},
        {"self_train": False},
        {"adaptive_alpha": False},
    ]
    for base_name, wf, wr, wb in core + [("kd_default", 0.0, 0.0, 0.0)]:
        for v in variants:
            vtag = "_".join(f"{k}{str(val).replace('.', 'p')}" for k, val in v.items())
            c = {"name": f"{base_name}_{vtag}", "w_feat": wf, "w_rel": wr, "w_branch": wb}
            c.update(v)
            cfgs.append(c)

    # deterministic dedupe by name while preserving order
    dedup = []
    seen = set()
    for c in cfgs:
        if c["name"] not in seen:
            dedup.append(c)
            seen.add(c["name"])
        if len(dedup) >= max_runs:
            break

    # merge with base kd defaults
    merged = []
    for c in dedup:
        m = dict(base_kd_cfg)
        m.update(c)
        merged.append(m)
    return merged


def _run_dual_kd_experiment_set(
    *,
    exp_name,
    kd_cfg_list,
    student_ctor,
    teacher,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    warmup_epochs,
    patience,
    lr,
    weight_decay,
    save_dir,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    best_global = None

    for i, kd_cfg in enumerate(kd_cfg_list, start=1):
        name = kd_cfg["name"]
        print(f"\n[{exp_name}] KD config {i}/{len(kd_cfg_list)}: {name}")
        student = student_ctor().to(device)

        proj_fused = None
        if kd_cfg.get("w_feat", 0.0) > 0.0 or kd_cfg.get("w_rel", 0.0) > 0.0:
            proj_fused = KDProjector(student.norm.normalized_shape[0], teacher.embed_dim).to(device)
            params = list(student.parameters()) + list(proj_fused.parameters())
        else:
            params = student.parameters()

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        sch = get_scheduler(opt, warmup_epochs, epochs)

        best_auc, best_state, best_state_proj, no_improve = 0.0, None, None, 0
        kd_active = not kd_cfg.get("adaptive_alpha", False)
        stable_count = 0
        prev_val_loss = None

        for ep in range(epochs):
            current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
            kd_cfg_ep = dict(kd_cfg)
            kd_cfg_ep["alpha_kd"] = current_alpha

            train_loss, train_auc = _train_kd_epoch_dual_advanced(
                student, teacher, train_loader, opt, device, kd_cfg_ep, proj_fused=proj_fused
            )
            val_auc, _, _ = evaluate_kd_dual(student, val_loader, device)
            sch.step()

            if not kd_active and kd_cfg.get("adaptive_alpha", False):
                val_loss = evaluate_bce_loss_dual(student, val_loader, device)
                if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                    stable_count += 1
                else:
                    stable_count = 0
                prev_val_loss = val_loss
                if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                    kd_active = True
                    print(f"[{name}] Activating KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                best_state_proj = (
                    {k: v.detach().cpu().clone() for k, v in proj_fused.state_dict().items()}
                    if proj_fused is not None
                    else None
                )
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 5 == 0:
                print(
                    f"[{name}] ep {ep+1}: train_auc={train_auc:.4f}, "
                    f"val_auc={val_auc:.4f}, best={best_auc:.4f} | alpha_kd={current_alpha:.2f}"
                )
            if no_improve >= patience:
                print(f"[{name}] Early stopping at epoch {ep+1}")
                break

        if best_state is not None:
            student.load_state_dict(best_state)
            if proj_fused is not None and best_state_proj is not None:
                proj_fused.load_state_dict(best_state_proj)

        if kd_cfg.get("self_train", False):
            print(f"[{name}] Self-train...")
            opt_st = torch.optim.AdamW(
                list(student.parameters()) + (list(proj_fused.parameters()) if proj_fused is not None else []),
                lr=kd_cfg["self_train_lr"],
            )
            best_auc_st = best_auc
            no_improve = 0
            for ep in range(kd_cfg["self_train_epochs"]):
                st_loss = self_train_student_dual(student, teacher, train_loader, opt_st, device, kd_cfg)
                val_auc, _, _ = evaluate_kd_dual(student, val_loader, device)
                if val_auc > best_auc_st:
                    best_auc_st = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                    best_state_proj = (
                        {k: v.detach().cpu().clone() for k, v in proj_fused.state_dict().items()}
                        if proj_fused is not None
                        else None
                    )
                    no_improve = 0
                else:
                    no_improve += 1
                if (ep + 1) % 2 == 0:
                    print(f"[{name}] self ep {ep+1}: loss={st_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_st:.4f}")
                if no_improve >= kd_cfg["self_train_patience"]:
                    break
            if best_state is not None:
                student.load_state_dict(best_state)
                if proj_fused is not None and best_state_proj is not None:
                    proj_fused.load_state_dict(best_state_proj)
            best_auc = max(best_auc, best_auc_st)

        auc_test, preds_test, _ = evaluate_kd_dual(student, test_loader, device)
        summary_rows.append(
            {
                "name": name,
                "val_auc": float(best_auc),
                "test_auc": float(auc_test),
                "alpha_kd": float(kd_cfg["alpha_kd"]),
                "temperature": float(kd_cfg["temperature"]),
                "w_feat": float(kd_cfg.get("w_feat", 0.0)),
                "w_rel": float(kd_cfg.get("w_rel", 0.0)),
                "w_branch": float(kd_cfg.get("w_branch", 0.0)),
                "conf_weighted": int(bool(kd_cfg.get("conf_weighted", True))),
                "adaptive_alpha": int(bool(kd_cfg.get("adaptive_alpha", True))),
                "self_train": int(bool(kd_cfg.get("self_train", True))),
            }
        )

        torch.save({"model": student.state_dict(), "val_auc": best_auc, "test_auc": auc_test, "cfg": kd_cfg}, save_dir / f"{name}.pt")
        with open(save_dir / f"{name}.json", "w") as f:
            json.dump(summary_rows[-1], f, indent=2)

        if best_global is None or best_auc > best_global["val_auc"]:
            best_global = {"name": name, "val_auc": best_auc, "test_auc": auc_test, "preds": preds_test, "model": student.state_dict()}

    # write summary csv
    with open(save_dir / "kd_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with open(save_dir / "kd_sweep_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"[{exp_name}] Best KD config: {best_global['name']} | val_auc={best_global['val_auc']:.4f} | test_auc={best_global['test_auc']:.4f}")
    return best_global["test_auc"], best_global["preds"], best_global["name"], summary_rows, best_global["model"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint dir with merge_count/unmerge_predictor (and optionally teacher/baseline).")
    parser.add_argument("--save_dir", type=str, default="checkpoints/unmerge_new_dualview_kd", help="Where to save new results.")
    parser.add_argument("--run_name", type=str, default="dualview_kd_only", help="Run name for output folder.")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--n_train_jets", type=int, default=200000)
    parser.add_argument("--offset_jets", type=int, default=200000, help="Skip this many jets to get a new 200k slice.")
    parser.add_argument("--max_constits", type=int, default=80)
    parser.add_argument("--max_merge_count", type=int, default=10)
    parser.add_argument(
        "--unmerge_head_mode",
        type=str,
        default="single",
        choices=["single", "two", "four"],
        help="Must match the unmerge_predictor checkpoint.",
    )
    parser.add_argument(
        "--unmerge_parent_mode",
        type=str,
        default="none",
        choices=["none", "query", "cross"],
    )
    parser.add_argument(
        "--unmerge_relpos_mode",
        type=str,
        default="none",
        choices=["none", "attn"],
    )
    parser.add_argument(
        "--unmerge_local_attn_mode",
        type=str,
        default="none",
        choices=["none", "soft", "hard"],
    )
    parser.add_argument("--unmerge_local_attn_radius", type=float, default=0.2)
    parser.add_argument("--unmerge_local_attn_scale", type=float, default=2.0)
    parser.add_argument(
        "--unmerge_target_mode",
        type=str,
        default="absolute",
        choices=["absolute", "normalized"],
    )
    parser.add_argument("--kd_sweep", action="store_true", help="Run KD sweep (~30 configs) for dual-view KD stages.")
    parser.add_argument("--kd_sweep_max", type=int, default=30, help="Maximum number of KD configs to run.")
    parser.add_argument(
        "--kd_sweep_target",
        type=str,
        default="dual_flag",
        choices=["dual", "dual_flag", "both"],
        help="Which KD stage(s) to sweep. Other stage(s) use default KD config.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ckpt_dir = Path(args.ckpt_dir)
    save_root = Path(args.save_dir) / args.run_name
    save_root.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_path = Path(args.train_path)
    if train_path.is_dir():
        train_files = sorted(train_path.glob("*.h5"))
    else:
        # allow comma-separated list of files
        train_files = [Path(p) for p in str(args.train_path).split(",") if p.strip()]
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {args.train_path}")

    max_jets_needed = args.offset_jets + args.n_train_jets
    all_data_full, all_labels_full, _, _, _ = utils.load_from_files(
        [str(p) for p in train_files],
        max_jets=max_jets_needed,
        max_constits=args.max_constits,
        use_train_weights=False,
    )
    if all_data_full.shape[0] < max_jets_needed:
        raise RuntimeError(
            f"Not enough jets for offset {args.offset_jets} + n_train_jets {args.n_train_jets}. "
            f"Got {all_data_full.shape[0]}."
        )
    all_data = all_data_full[args.offset_jets:args.offset_jets + args.n_train_jets]
    all_labels = all_labels_full[args.offset_jets:args.offset_jets + args.n_train_jets]
    all_labels = all_labels.astype(np.int64)

    eta = all_data[:, :, ETA_IDX].astype(np.float32)
    phi = all_data[:, :, PHI_IDX].astype(np.float32)
    pt = all_data[:, :, PT_IDX].astype(np.float32)
    mask_raw = pt > 0
    E = pt * np.cosh(np.clip(eta, -5, 5))
    const_off = np.stack([pt, eta, phi, E], axis=-1).astype(np.float32)
    masks_off = mask_raw

    print("Applying HLT effects...")
    hlt_const, hlt_mask, origin_counts, origin_lists, _ = apply_hlt_effects_with_tracking(
        const_off, masks_off, CONFIG, seed=RANDOM_SEED
    )

    features_off = compute_features(const_off, masks_off)
    features_hlt = compute_features(hlt_const, hlt_mask)

    idx = np.arange(len(all_labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=all_labels[temp_idx])

    feat_means, feat_stds = get_stats(features_off, masks_off, train_idx)
    features_off_std = standardize(features_off, masks_off, feat_means, feat_stds)
    features_hlt_std = standardize(features_hlt, hlt_mask, feat_means, feat_stds)

    # Train teacher/baseline on the new 200k jets
    BS = CONFIG["training"]["batch_size"]
    train_ds_off = JetDataset(features_off_std[train_idx], masks_off[train_idx], all_labels[train_idx])
    val_ds_off = JetDataset(features_off_std[val_idx], masks_off[val_idx], all_labels[val_idx])
    test_ds_off = JetDataset(features_off_std[test_idx], masks_off[test_idx], all_labels[test_idx])
    train_loader_off = torch.utils.data.DataLoader(train_ds_off, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_off = torch.utils.data.DataLoader(val_ds_off, batch_size=BS, shuffle=False)
    test_loader_off = torch.utils.data.DataLoader(test_ds_off, batch_size=BS, shuffle=False)

    teacher = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_t = get_scheduler(opt_t, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_t, best_state_t, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(teacher, train_loader_off, opt_t, device)
        val_auc, _, _ = eval_classifier(teacher, val_loader_off, device)
        sch_t.step()
        if val_auc > best_auc_t:
            best_auc_t = val_auc
            best_state_t = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Teacher ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_t:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping teacher at epoch {ep+1}")
            break
    if best_state_t is not None:
        teacher.load_state_dict(best_state_t)

    # Baseline (HLT)
    train_ds_hlt = JetDataset(features_hlt_std[train_idx], hlt_mask[train_idx], all_labels[train_idx])
    val_ds_hlt = JetDataset(features_hlt_std[val_idx], hlt_mask[val_idx], all_labels[val_idx])
    test_ds_hlt = JetDataset(features_hlt_std[test_idx], hlt_mask[test_idx], all_labels[test_idx])
    train_loader_hlt = torch.utils.data.DataLoader(train_ds_hlt, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_hlt = torch.utils.data.DataLoader(val_ds_hlt, batch_size=BS, shuffle=False)
    test_loader_hlt = torch.utils.data.DataLoader(test_ds_hlt, batch_size=BS, shuffle=False)

    baseline = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_b = get_scheduler(opt_b, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_b, best_state_b, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(baseline, train_loader_hlt, opt_b, device)
        val_auc, _, _ = eval_classifier(baseline, val_loader_hlt, device)
        sch_b.step()
        if val_auc > best_auc_b:
            best_auc_b = val_auc
            best_state_b = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Baseline ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_b:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping baseline at epoch {ep+1}")
            break
    if best_state_b is not None:
        baseline.load_state_dict(best_state_b)

    auc_teacher, preds_teacher, labs = eval_classifier(teacher, test_loader_off, device)
    auc_baseline, preds_baseline, _ = eval_classifier(baseline, test_loader_hlt, device)

    # HLT + KD (teacher on offline, student on HLT)
    kd_train_ds_hlt = UnmergeKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    kd_val_ds_hlt = UnmergeKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    kd_test_ds_hlt = UnmergeKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    kd_train_loader_hlt = torch.utils.data.DataLoader(kd_train_ds_hlt, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader_hlt = torch.utils.data.DataLoader(kd_val_ds_hlt, batch_size=BS, shuffle=False)
    kd_test_loader_hlt = torch.utils.data.DataLoader(kd_test_ds_hlt, batch_size=BS, shuffle=False)

    hlt_kd = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_hlt_kd = torch.optim.AdamW(hlt_kd.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_hlt_kd = get_scheduler(opt_hlt_kd, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])

    kd_cfg = CONFIG["kd"]
    best_auc_hlt_kd, best_state_hlt_kd, no_improve = 0.0, None, 0
    kd_active = not kd_cfg["adaptive_alpha"]
    stable_count = 0
    prev_val_loss = None

    for ep in range(CONFIG["training"]["epochs"]):
        current_alpha = kd_cfg["alpha_kd"] if kd_active else 0.0
        kd_cfg_ep = dict(kd_cfg)
        kd_cfg_ep["alpha_kd"] = current_alpha

        train_loss, train_auc = train_kd_epoch(hlt_kd, teacher, kd_train_loader_hlt, opt_hlt_kd, device, kd_cfg_ep)
        val_auc, _, _ = evaluate_kd(hlt_kd, kd_val_loader_hlt, device)
        sch_hlt_kd.step()

        if not kd_active and kd_cfg["adaptive_alpha"]:
            val_loss = evaluate_bce_loss_unmerged(hlt_kd, kd_val_loader_hlt, device)
            if prev_val_loss is not None and abs(prev_val_loss - val_loss) < kd_cfg["alpha_stable_delta"]:
                stable_count += 1
            else:
                stable_count = 0
            prev_val_loss = val_loss
            if ep + 1 >= kd_cfg["alpha_warmup_min_epochs"] and stable_count >= kd_cfg["alpha_stable_patience"]:
                kd_active = True
                print(f"Activating HLT KD ramp at epoch {ep+1} (val_loss={val_loss:.4f})")

        if val_auc > best_auc_hlt_kd:
            best_auc_hlt_kd = val_auc
            best_state_hlt_kd = {k: v.detach().cpu().clone() for k, v in hlt_kd.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 5 == 0:
            print(
                f"HLT+KD ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, "
                f"best={best_auc_hlt_kd:.4f} | alpha_kd={current_alpha:.2f}"
            )
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping HLT+KD student at epoch {ep+1}")
            break

    if best_state_hlt_kd is not None:
        hlt_kd.load_state_dict(best_state_hlt_kd)

    if kd_cfg["self_train"]:
        print("\nSelf-train HLT+KD...")
        opt_st = torch.optim.AdamW(hlt_kd.parameters(), lr=kd_cfg["self_train_lr"])
        best_auc_st = best_auc_hlt_kd
        no_improve = 0
        for ep in range(kd_cfg["self_train_epochs"]):
            st_loss = self_train_student(hlt_kd, teacher, kd_train_loader_hlt, opt_st, device, kd_cfg)
            val_auc, _, _ = evaluate_kd(hlt_kd, kd_val_loader_hlt, device)
            if val_auc > best_auc_st:
                best_auc_st = val_auc
                best_state_hlt_kd = {k: v.detach().cpu().clone() for k, v in hlt_kd.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if (ep + 1) % 2 == 0:
                print(f"Self ep {ep+1}: loss={st_loss:.4f}, val_auc={val_auc:.4f}, best={best_auc_st:.4f}")
            if no_improve >= kd_cfg["self_train_patience"]:
                break
        if best_state_hlt_kd is not None:
            hlt_kd.load_state_dict(best_state_hlt_kd)

    auc_hlt_kd, preds_hlt_kd, _ = evaluate_kd(hlt_kd, kd_test_loader_hlt, device)

    # Load merge-count predictor
    max_count = args.max_merge_count
    count_model = MergeCountPredictor(input_dim=7, num_classes=max_count, **CONFIG["merge_count_model"]).to(device)
    count_ckpt = torch.load(ckpt_dir / "merge_count.pt", map_location=device)
    count_model.load_state_dict(count_ckpt["model"])
    BS_cnt = CONFIG["merge_count_training"]["batch_size"]
    pred_counts = predict_counts(count_model, features_hlt_std, hlt_mask, BS_cnt, device, max_count)

    # Load unmerge predictor (auto-match relpos/local-attn to checkpoint)
    unmerge_ckpt = torch.load(ckpt_dir / "unmerge_predictor.pt", map_location=device)
    ckpt_keys = list(unmerge_ckpt["model"].keys())
    has_encoder_layers = any(k.startswith("encoder_layers.") for k in ckpt_keys)
    has_relpos_mlp = any(k.startswith("relpos_mlp.") for k in ckpt_keys)
    has_std_encoder = any(k.startswith("encoder.layers.") for k in ckpt_keys)

    relpos_mode = args.unmerge_relpos_mode
    local_attn_mode = args.unmerge_local_attn_mode
    if has_std_encoder:
        # checkpoint is the vanilla transformer encoder
        if relpos_mode != "none" or local_attn_mode != "none":
            print("Info: checkpoint uses standard encoder; forcing relpos/local_attn to none for loading.")
            relpos_mode = "none"
            local_attn_mode = "none"
    elif has_encoder_layers:
        # checkpoint uses custom relpos/local-attn encoder
        if has_relpos_mlp and relpos_mode == "none":
            print("Info: checkpoint has relpos_mlp; forcing relpos_mode='attn' for loading.")
            relpos_mode = "attn"
        if not has_relpos_mlp and relpos_mode == "none" and local_attn_mode == "none":
            # need encoder_layers path; local attn has no params, so soft is safest
            print("Info: checkpoint uses relpos/local-attn encoder; forcing local_attn_mode='soft' for loading.")
            local_attn_mode = "soft"

    unmerge_model = UnmergePredictor(
        input_dim=7,
        max_count=max_count,
        head_mode=args.unmerge_head_mode,
        parent_mode=args.unmerge_parent_mode,
        relpos_mode=relpos_mode,
        local_attn_mode=local_attn_mode,
        local_attn_radius=args.unmerge_local_attn_radius,
        local_attn_scale=args.unmerge_local_attn_scale,
        **CONFIG["unmerge_model"],
    ).to(device)
    unmerge_model.load_state_dict(unmerge_ckpt["model"])

    # Compute target mean/std on train split (same logic as training)
    count_label = np.clip(origin_counts, 1, max_count) - 1
    samples = []
    for j in range(len(all_labels)):
        for idx_tok in range(args.max_constits):
            origin = origin_lists[j][idx_tok]
            if hlt_mask[j, idx_tok] and len(origin) > 1:
                if len(origin) > max_count:
                    continue
                pc = int(pred_counts[j, idx_tok])
                if pc < 2:
                    pc = 2
                if pc > max_count:
                    pc = max_count
                samples.append((j, idx_tok, origin, pc))

    train_idx_set = set(train_idx)
    train_samples = [s for s in samples if s[0] in train_idx_set]
    if len(train_samples) == 0:
        raise RuntimeError("No merged samples in training split.")

    train_targets = []
    for s in train_samples:
        target_abs = const_off[s[0], s[2], :4]
        if args.unmerge_target_mode == "normalized":
            parent = hlt_const[s[0], s[1], :4].astype(np.float32)
            pt_p = max(parent[0], 1e-8)
            eta_p = parent[1]
            phi_p = parent[2]
            e_p = max(parent[3], 1e-8)
            pt_frac = target_abs[:, 0] / pt_p
            e_frac = target_abs[:, 3] / e_p
            deta = target_abs[:, 1] - eta_p
            dphi = np.arctan2(np.sin(target_abs[:, 2] - phi_p), np.cos(target_abs[:, 2] - phi_p))
            target = np.stack([pt_frac, deta, dphi, e_frac], axis=-1)
            train_targets.append(target.astype(np.float32))
        else:
            train_targets.append(target_abs.astype(np.float32))
    flat_train = np.concatenate(train_targets, axis=0)
    tgt_mean = flat_train.mean(axis=0)
    tgt_std = flat_train.std(axis=0) + 1e-8

    # Build unmerged dataset from pretrained unmerger
    unmerged_const, unmerged_mask, _ = build_unmerged_dataset(
        features_hlt_std,
        hlt_mask,
        hlt_const,
        pred_counts,
        unmerge_model,
        tgt_mean,
        tgt_std,
        max_count,
        args.max_constits,
        device,
        CONFIG["unmerge_training"]["batch_size"],
        args.unmerge_target_mode,
    )
    features_unmerged = compute_features(unmerged_const, unmerged_mask)
    features_unmerged_std = standardize(features_unmerged, unmerged_mask, feat_means, feat_stds)

    # Build unmerged dataset
    unmerged_const, unmerged_mask, unmerged_flag = build_unmerged_dataset(
        features_hlt_std,
        hlt_mask,
        hlt_const,
        pred_counts,
        unmerge_model,
        tgt_mean,
        tgt_std,
        max_count,
        args.max_constits,
        device,
        CONFIG["unmerge_training"]["batch_size"],
        args.unmerge_target_mode,
    )
    features_unmerged = compute_features(unmerged_const, unmerged_mask)
    features_unmerged_std = standardize(features_unmerged, unmerged_mask, feat_means, feat_stds)
    features_unmerged_flag = np.concatenate(
        [features_unmerged_std, unmerged_flag[..., None]], axis=-1
    ).astype(np.float32)

    # Unmerge classifier (single-view)
    train_ds_um = JetDataset(features_unmerged_std[train_idx], unmerged_mask[train_idx], all_labels[train_idx])
    val_ds_um = JetDataset(features_unmerged_std[val_idx], unmerged_mask[val_idx], all_labels[val_idx])
    test_ds_um = JetDataset(features_unmerged_std[test_idx], unmerged_mask[test_idx], all_labels[test_idx])
    train_loader_um = torch.utils.data.DataLoader(train_ds_um, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_um = torch.utils.data.DataLoader(val_ds_um, batch_size=BS, shuffle=False)
    test_loader_um = torch.utils.data.DataLoader(test_ds_um, batch_size=BS, shuffle=False)

    unmerge_cls = ParticleTransformer(input_dim=7, **CONFIG["model"]).to(device)
    opt_u = torch.optim.AdamW(unmerge_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_u = get_scheduler(opt_u, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_u, best_state_u, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(unmerge_cls, train_loader_um, opt_u, device)
        val_auc, _, _ = eval_classifier(unmerge_cls, val_loader_um, device)
        sch_u.step()
        if val_auc > best_auc_u:
            best_auc_u = val_auc
            best_state_u = {k: v.detach().cpu().clone() for k, v in unmerge_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Unmerge ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_u:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping unmerge classifier at epoch {ep+1}")
            break
    if best_state_u is not None:
        unmerge_cls.load_state_dict(best_state_u)
    auc_unmerge, preds_unmerge, _ = eval_classifier(unmerge_cls, test_loader_um, device)

    # Unmerge + MergeFlag classifier
    train_ds_umf = JetDataset(features_unmerged_flag[train_idx], unmerged_mask[train_idx], all_labels[train_idx])
    val_ds_umf = JetDataset(features_unmerged_flag[val_idx], unmerged_mask[val_idx], all_labels[val_idx])
    test_ds_umf = JetDataset(features_unmerged_flag[test_idx], unmerged_mask[test_idx], all_labels[test_idx])
    train_loader_umf = torch.utils.data.DataLoader(train_ds_umf, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_umf = torch.utils.data.DataLoader(val_ds_umf, batch_size=BS, shuffle=False)
    test_loader_umf = torch.utils.data.DataLoader(test_ds_umf, batch_size=BS, shuffle=False)

    unmerge_flag_cls = ParticleTransformer(input_dim=8, **CONFIG["model"]).to(device)
    opt_uf = torch.optim.AdamW(unmerge_flag_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_uf = get_scheduler(opt_uf, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_uf, best_state_uf, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        _, train_auc = train_classifier(unmerge_flag_cls, train_loader_umf, opt_uf, device)
        val_auc, _, _ = eval_classifier(unmerge_flag_cls, val_loader_umf, device)
        sch_uf.step()
        if val_auc > best_auc_uf:
            best_auc_uf = val_auc
            best_state_uf = {k: v.detach().cpu().clone() for k, v in unmerge_flag_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Unmerge+MF ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_uf:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping unmerge+MF classifier at epoch {ep+1}")
            break
    if best_state_uf is not None:
        unmerge_flag_cls.load_state_dict(best_state_uf)
    auc_unmerge_flag, preds_unmerge_flag, _ = eval_classifier(unmerge_flag_cls, test_loader_umf, device)

    # Dual-view datasets
    train_ds_dual = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_std[train_idx],
        unmerged_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    val_ds_dual = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_std[val_idx],
        unmerged_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    test_ds_dual = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_std[test_idx],
        unmerged_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    train_loader_dual = torch.utils.data.DataLoader(train_ds_dual, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_dual = torch.utils.data.DataLoader(val_ds_dual, batch_size=BS, shuffle=False)
    test_loader_dual = torch.utils.data.DataLoader(test_ds_dual, batch_size=BS, shuffle=False)

    # Dual-view classifier
    dual_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    opt_dv = torch.optim.AdamW(dual_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_dv = get_scheduler(opt_dv, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_dv, best_state_dv, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        kd_cfg_zero = dict(CONFIG["kd"])
        kd_cfg_zero["alpha_kd"] = 0.0
        _, train_auc = train_kd_epoch_dual(dual_cls, teacher, train_loader_dual, opt_dv, device, kd_cfg_zero)
        val_auc, _, _ = evaluate_kd_dual(dual_cls, val_loader_dual, device)
        sch_dv.step()
        if val_auc > best_auc_dv:
            best_auc_dv = val_auc
            best_state_dv = {k: v.detach().cpu().clone() for k, v in dual_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"DualView ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_dv:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping dual-view classifier at epoch {ep+1}")
            break
    if best_state_dv is not None:
        dual_cls.load_state_dict(best_state_dv)
    auc_dual, preds_dual, _ = evaluate_kd_dual(dual_cls, test_loader_dual, device)

    # Dual-view + merge-flag classifier (train with KD dataset that includes flag)
    train_ds_dual_flag = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_flag[train_idx],
        unmerged_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    val_ds_dual_flag = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_flag[val_idx],
        unmerged_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    test_ds_dual_flag = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_flag[test_idx],
        unmerged_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    train_loader_dual_flag = torch.utils.data.DataLoader(train_ds_dual_flag, batch_size=BS, shuffle=True, drop_last=True)
    val_loader_dual_flag = torch.utils.data.DataLoader(val_ds_dual_flag, batch_size=BS, shuffle=False)
    test_loader_dual_flag = torch.utils.data.DataLoader(test_ds_dual_flag, batch_size=BS, shuffle=False)

    dual_flag_cls = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]).to(device)
    opt_dvf = torch.optim.AdamW(dual_flag_cls.parameters(), lr=CONFIG["training"]["lr"], weight_decay=CONFIG["training"]["weight_decay"])
    sch_dvf = get_scheduler(opt_dvf, CONFIG["training"]["warmup_epochs"], CONFIG["training"]["epochs"])
    best_auc_dvf, best_state_dvf, no_improve = 0.0, None, 0
    for ep in range(CONFIG["training"]["epochs"]):
        kd_cfg_zero = dict(CONFIG["kd"])
        kd_cfg_zero["alpha_kd"] = 0.0
        _, train_auc = train_kd_epoch_dual(dual_flag_cls, teacher, train_loader_dual_flag, opt_dvf, device, kd_cfg_zero)
        val_auc, _, _ = evaluate_kd_dual(dual_flag_cls, val_loader_dual_flag, device)
        sch_dvf.step()
        if val_auc > best_auc_dvf:
            best_auc_dvf = val_auc
            best_state_dvf = {k: v.detach().cpu().clone() for k, v in dual_flag_cls.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"DualView+MF ep {ep+1}: train_auc={train_auc:.4f}, val_auc={val_auc:.4f}, best={best_auc_dvf:.4f}")
        if no_improve >= CONFIG["training"]["patience"]:
            print(f"Early stopping dual-view+MF classifier at epoch {ep+1}")
            break
    if best_state_dvf is not None:
        dual_flag_cls.load_state_dict(best_state_dvf)
    auc_dual_flag, preds_dual_flag, _ = evaluate_kd_dual(dual_flag_cls, test_loader_dual_flag, device)

    # KD datasets
    kd_train_ds = DualViewKDDataset(
        features_hlt_std[train_idx],
        hlt_mask[train_idx],
        features_unmerged_std[train_idx],
        unmerged_mask[train_idx],
        features_off_std[train_idx],
        masks_off[train_idx],
        all_labels[train_idx],
    )
    kd_val_ds = DualViewKDDataset(
        features_hlt_std[val_idx],
        hlt_mask[val_idx],
        features_unmerged_std[val_idx],
        unmerged_mask[val_idx],
        features_off_std[val_idx],
        masks_off[val_idx],
        all_labels[val_idx],
    )
    kd_test_ds = DualViewKDDataset(
        features_hlt_std[test_idx],
        hlt_mask[test_idx],
        features_unmerged_std[test_idx],
        unmerged_mask[test_idx],
        features_off_std[test_idx],
        masks_off[test_idx],
        all_labels[test_idx],
    )
    kd_train_loader = torch.utils.data.DataLoader(kd_train_ds, batch_size=BS, shuffle=True, drop_last=True)
    kd_val_loader = torch.utils.data.DataLoader(kd_val_ds, batch_size=BS, shuffle=False)
    kd_test_loader = torch.utils.data.DataLoader(kd_test_ds, batch_size=BS, shuffle=False)

    base_kd_cfg = dict(CONFIG["kd"])
    base_kd_cfg.update({"name": "kd_default", "w_feat": 0.0, "w_rel": 0.0, "w_branch": 0.0})
    sweep_cfgs = _build_kd_sweep_configs(base_kd_cfg, max_runs=args.kd_sweep_max) if args.kd_sweep else [base_kd_cfg]

    dual_should_sweep = args.kd_sweep and args.kd_sweep_target in ("dual", "both")
    dual_flag_should_sweep = args.kd_sweep and args.kd_sweep_target in ("dual_flag", "both")

    # DualView + KD
    dual_cfgs = sweep_cfgs if dual_should_sweep else [base_kd_cfg]
    auc_dual_kd, preds_dual_kd, best_dual_name, dual_summary, best_dual_state = _run_dual_kd_experiment_set(
        exp_name="DualView+KD",
        kd_cfg_list=dual_cfgs,
        student_ctor=lambda: DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]),
        teacher=teacher,
        train_loader=kd_train_loader,
        val_loader=kd_val_loader,
        test_loader=kd_test_loader,
        device=device,
        epochs=CONFIG["training"]["epochs"],
        warmup_epochs=CONFIG["training"]["warmup_epochs"],
        patience=CONFIG["training"]["patience"],
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        save_dir=save_root / "kd_sweep_dual",
    )

    # DualView + MergeFlag + KD
    dual_flag_cfgs = sweep_cfgs if dual_flag_should_sweep else [base_kd_cfg]
    auc_dual_flag_kd, preds_dual_flag_kd, best_dual_flag_name, dual_flag_summary, best_dual_flag_state = _run_dual_kd_experiment_set(
        exp_name="DualView+MF+KD",
        kd_cfg_list=dual_flag_cfgs,
        student_ctor=lambda: DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]),
        teacher=teacher,
        train_loader=train_loader_dual_flag,
        val_loader=val_loader_dual_flag,
        test_loader=test_loader_dual_flag,
        device=device,
        epochs=CONFIG["training"]["epochs"],
        warmup_epochs=CONFIG["training"]["warmup_epochs"],
        patience=CONFIG["training"]["patience"],
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        save_dir=save_root / "kd_sweep_dual_flag",
    )

    kd_student = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=7, **CONFIG["model"]).to(device)
    kd_student.load_state_dict(best_dual_state)
    kd_student_flag = DualViewCrossAttnClassifier(input_dim_a=7, input_dim_b=8, **CONFIG["model"]).to(device)
    kd_student_flag.load_state_dict(best_dual_flag_state)
    print(f"Selected DualView+KD config: {best_dual_name}")
    print(f"Selected DualView+MF+KD config: {best_dual_flag_name}")

    print("\nFINAL TEST EVALUATION")
    print(f"Teacher (Offline) AUC: {auc_teacher:.4f}")
    print(f"Baseline (HLT)   AUC: {auc_baseline:.4f}")
    print(f"HLT+KD           AUC: {auc_hlt_kd:.4f}")
    print(f"Unmerge          AUC: {auc_unmerge:.4f}")
    print(f"Unmerge+MF       AUC: {auc_unmerge_flag:.4f}")
    print(f"Dual-View        AUC: {auc_dual:.4f}")
    print(f"Dual-View+MF     AUC: {auc_dual_flag:.4f}")
    print(f"Dual-View+KD     AUC: {auc_dual_kd:.4f}")
    print(f"Dual-View+MF+KD  AUC: {auc_dual_flag_kd:.4f}")

    # Plot teacher/baseline + all models
    fpr_t, tpr_t, _ = roc_curve(labs, preds_teacher)
    fpr_b, tpr_b, _ = roc_curve(labs, preds_baseline)
    fpr_b_kd, tpr_b_kd, _ = roc_curve(labs, preds_hlt_kd)
    fpr_u, tpr_u, _ = roc_curve(labs, preds_unmerge)
    fpr_uf, tpr_uf, _ = roc_curve(labs, preds_unmerge_flag)
    fpr_dv, tpr_dv, _ = roc_curve(labs, preds_dual)
    fpr_dvf, tpr_dvf, _ = roc_curve(labs, preds_dual_flag)
    fpr_dv_k, tpr_dv_k, _ = roc_curve(labs, preds_dual_kd)
    fpr_dvf_k, tpr_dvf_k, _ = roc_curve(labs, preds_dual_flag_kd)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"HLT Baseline (AUC={auc_baseline:.3f})", color="steelblue", linewidth=2)
    plt.plot(tpr_b_kd, fpr_b_kd, "--", label=f"HLT+KD (AUC={auc_hlt_kd:.3f})", color="royalblue", linewidth=2)
    plt.plot(tpr_u, fpr_u, ":", label=f"Unmerge (AUC={auc_unmerge:.3f})", color="forestgreen", linewidth=2)
    plt.plot(tpr_uf, fpr_uf, "-.", label=f"Unmerge+MF (AUC={auc_unmerge_flag:.3f})", color="darkorange", linewidth=2)
    plt.plot(tpr_dv, fpr_dv, "-", label=f"DualView (AUC={auc_dual:.3f})", color="teal", linewidth=2)
    plt.plot(tpr_dvf, fpr_dvf, "--", label=f"DualView+MF (AUC={auc_dual_flag:.3f})", color="orchid", linewidth=2)
    plt.plot(tpr_dv_k, fpr_dv_k, ":", label=f"DualView+KD (AUC={auc_dual_kd:.3f})", color="slateblue", linewidth=2)
    plt.plot(tpr_dvf_k, fpr_dvf_k, "-.", label=f"DualView+MF+KD (AUC={auc_dual_flag_kd:.3f})", color="darkslateblue", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "results_all.png", dpi=300)
    plt.close()

    # Also save teacher+baseline+dualview+kd (quick view)
    plt.figure(figsize=(8, 6))
    plt.plot(tpr_t, fpr_t, "-", label=f"Teacher (AUC={auc_teacher:.3f})", color="crimson", linewidth=2)
    plt.plot(tpr_b, fpr_b, "--", label=f"HLT Baseline (AUC={auc_baseline:.3f})", color="steelblue", linewidth=2)
    plt.plot(tpr_b_kd, fpr_b_kd, "--", label=f"HLT+KD (AUC={auc_hlt_kd:.3f})", color="royalblue", linewidth=2)
    plt.plot(tpr_dv_k, fpr_dv_k, ":", label=f"DualView+KD (AUC={auc_dual_kd:.3f})", color="slateblue", linewidth=2)
    plt.ylabel("False Positive Rate", fontsize=12)
    plt.xlabel("True Positive Rate (Signal efficiency)", fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_root / "results_teacher_baseline_dualview_kd.png", dpi=300)
    plt.close()

    np.savez(
        save_root / "results.npz",
        kd_sweep=int(args.kd_sweep),
        kd_sweep_target=args.kd_sweep_target,
        best_dual_kd_name=best_dual_name,
        best_dual_flag_kd_name=best_dual_flag_name,
        auc_teacher=auc_teacher,
        auc_baseline=auc_baseline,
        auc_hlt_kd=auc_hlt_kd,
        auc_unmerge=auc_unmerge,
        auc_unmerge_flag=auc_unmerge_flag,
        auc_dual=auc_dual,
        auc_dual_flag=auc_dual_flag,
        auc_dual_kd=auc_dual_kd,
        auc_dual_flag_kd=auc_dual_flag_kd,
        fpr_teacher=fpr_t,
        tpr_teacher=tpr_t,
        fpr_baseline=fpr_b,
        tpr_baseline=tpr_b,
        fpr_hlt_kd=fpr_b_kd,
        tpr_hlt_kd=tpr_b_kd,
        fpr_unmerge=fpr_u,
        tpr_unmerge=tpr_u,
        fpr_unmerge_flag=fpr_uf,
        tpr_unmerge_flag=tpr_uf,
        fpr_dual=fpr_dv,
        tpr_dual=tpr_dv,
        fpr_dual_flag=fpr_dvf,
        tpr_dual_flag=tpr_dvf,
        fpr_dual_kd=fpr_dv_k,
        tpr_dual_kd=tpr_dv_k,
        fpr_dual_flag_kd=fpr_dvf_k,
        tpr_dual_flag_kd=tpr_dvf_k,
    )

    torch.save({"model": unmerge_cls.state_dict(), "auc": auc_unmerge}, save_root / "unmerge_classifier.pt")
    torch.save({"model": unmerge_flag_cls.state_dict(), "auc": auc_unmerge_flag}, save_root / "unmerge_mergeflag_classifier.pt")
    torch.save({"model": dual_cls.state_dict(), "auc": auc_dual}, save_root / "dual_view_classifier.pt")
    torch.save({"model": dual_flag_cls.state_dict(), "auc": auc_dual_flag}, save_root / "dual_view_mergeflag_classifier.pt")
    torch.save({"model": kd_student.state_dict(), "auc": auc_dual_kd}, save_root / "dual_view_kd.pt")
    torch.save({"model": kd_student_flag.state_dict(), "auc": auc_dual_flag_kd}, save_root / "dual_view_mergeflag_kd.pt")
    torch.save({"model": hlt_kd.state_dict(), "auc": auc_hlt_kd}, save_root / "hlt_kd.pt")
    torch.save({"model": teacher.state_dict(), "auc": auc_teacher}, save_root / "teacher.pt")
    torch.save({"model": baseline.state_dict(), "auc": auc_baseline}, save_root / "baseline.pt")

    print(f"\nSaved results to: {save_root}")


if __name__ == "__main__":
    main()
