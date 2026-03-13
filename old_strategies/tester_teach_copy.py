#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EFN Teacher-Student Knowledge Distillation (Unsmeared teacher, Smeared student)

Drop-in script based on your current file:
- Keeps your data loading, smearing, splitting, paired dataset, checkpoint directory, and checkpoint filenames.
- Main change: implements KD properly (train teacher first, freeze, then train student with KD).

Notes:
- No weights are used anywhere (as requested).
- EFN now outputs LOGITS internally (no sigmoid in forward). Accuracy still uses sigmoid + 0.5 threshold.
- KD loss matches your transformer notebook style: BCE(sigmoid(z_s/T), sigmoid(z_t/T)) * T^2
"""

from pathlib import Path
import argparse
import numpy as np
import random
import copy

import matplotlib.pyplot as plt  # kept because your original imports it

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from smearing import smear_dataset
import utils


# ----------------------------- Reproducibility ----------------------------- #
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------- Dataset ----------------------------- #
class PairedEFN_Dataset(Dataset):
    """Pairs unsmeared and smeared versions of the same jet for student-teacher training."""
    def __init__(self, pt_unsmeared, angular_unsmeared, pt_smeared, angular_smeared, labels):
        self.pt_unsmeared = torch.as_tensor(pt_unsmeared, dtype=torch.float32)
        self.angular_unsmeared = torch.as_tensor(angular_unsmeared, dtype=torch.float32)
        self.pt_smeared = torch.as_tensor(pt_smeared, dtype=torch.float32)
        self.angular_smeared = torch.as_tensor(angular_smeared, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        unsmeared = (self.pt_unsmeared[idx], self.angular_unsmeared[idx])
        smeared = (self.pt_smeared[idx], self.angular_smeared[idx])
        return unsmeared, smeared, self.labels[idx]


# ----------------------------- EFN Model ----------------------------- #
def mlp(sizes, last_act=None, dropout=0.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < (len(sizes) - 2):
            layers += [nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
    if last_act is not None:
        layers += [last_act]
    return nn.Sequential(*layers)


class EFN(nn.Module):
    """
    EFN that returns logits (recommended for KD + stable BCEWithLogits).
    Input: x = (pt, angular)
      pt: (B, N)
      angular: (B, N, 2)
    Output:
      logits: (B, 1)
    """
    def __init__(
        self,
        input_dim=2,
        phi_layers=(350, 350, 350, 350, 350),
        f_layers=(300, 300, 300, 300, 300),
        latent_dropout=0.084,
        f_dropouts=0.036,
        output_dim=1,
    ):
        super().__init__()
        K = phi_layers[-1]
        self.phi = mlp([input_dim] + list(phi_layers), dropout=0.0)
        self.latent_dropout = nn.Dropout(p=latent_dropout) if latent_dropout > 0 else nn.Identity()
        self.F = mlp([K] + list(f_layers) + [output_dim], dropout=f_dropouts)

    def forward(self, x):
        pt, ang = x
        ang_adjusted = self.phi(ang)          # (B, N, K)
        weighted = pt.unsqueeze(-1) * ang_adjusted
        summed = torch.sum(weighted, dim=1)  # (B, K)
        summed = self.latent_dropout(summed)
        logits = self.F(summed)              # (B, 1)
        return logits


# ----------------------------- KD Loss ----------------------------- #
def kd_loss_binary(student_logits, teacher_logits, T: float):
    """
    Binary KD (matches your transformer notebook style):
      BCE(sigmoid(z_s/T), sigmoid(z_t/T)) * T^2
    """
    s = student_logits.view(-1)
    t = teacher_logits.view(-1)
    s_soft = torch.sigmoid(s / T)
    t_soft = torch.sigmoid(t / T)
    return F.binary_cross_entropy(s_soft, t_soft) * (T ** 2)


# ----------------------------- Train / Eval Helpers ----------------------------- #
def batch_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.view(-1))
    preds = (probs >= 0.5).float()
    return (preds == labels.view(-1)).float().mean().item()


@torch.no_grad()
def eval_teacher(model, loader, device):
    model.eval()
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for unsmeared, smeared, labels in loader:
        pt_u, ang_u = unsmeared
        pt_u, ang_u = pt_u.to(device), ang_u.to(device)
        labels = labels.to(device)

        logits = model((pt_u, ang_u))
        loss = bce_logits(logits.view(-1), labels.view(-1))
        acc = batch_accuracy_from_logits(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_student_independent_epoch(model, loader, opt, device):
    """Train untaught student on smeared data only (no KD, just hard labels)"""
    model.train()
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for unsmeared, smeared, labels in loader:
        pt_s, ang_s = smeared
        pt_s, ang_s = pt_s.to(device), ang_s.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits = model((pt_s, ang_s))
        loss = bce_logits(logits.view(-1), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        acc = batch_accuracy_from_logits(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def eval_student_independent(model, loader, device):
    """Evaluate untaught student on smeared data"""
    model.eval()
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for unsmeared, smeared, labels in loader:
        pt_s, ang_s = smeared
        pt_s, ang_s = pt_s.to(device), ang_s.to(device)
        labels = labels.to(device)

        logits = model((pt_s, ang_s))
        loss = bce_logits(logits.view(-1), labels.view(-1))
        acc = batch_accuracy_from_logits(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def eval_student_kd(student, teacher, loader, device, T, alpha_kd):
    student.eval()
    teacher.eval()
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for unsmeared, smeared, labels in loader:
        pt_u, ang_u = unsmeared
        pt_s, ang_s = smeared

        pt_u, ang_u = pt_u.to(device), ang_u.to(device)
        pt_s, ang_s = pt_s.to(device), ang_s.to(device)
        labels = labels.to(device)

        t_logits = teacher((pt_u, ang_u))
        s_logits = student((pt_s, ang_s))

        hard = bce_logits(s_logits.view(-1), labels.view(-1))
        kd = kd_loss_binary(s_logits, t_logits, T)

        loss = alpha_kd * kd + (1.0 - alpha_kd) * hard
        acc = batch_accuracy_from_logits(s_logits, labels)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_teacher_epoch(model, loader, opt, device):
    model.train()
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for unsmeared, smeared, labels in loader:
        pt_u, ang_u = unsmeared
        pt_u, ang_u = pt_u.to(device), ang_u.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits = model((pt_u, ang_u))
        loss = bce_logits(logits.view(-1), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        acc = batch_accuracy_from_logits(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_student_kd_epoch(student, teacher, loader, opt, device, T, alpha_kd):
    student.train()
    teacher.eval()
    bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for unsmeared, smeared, labels in loader:
        pt_u, ang_u = unsmeared
        pt_s, ang_s = smeared

        pt_u, ang_u = pt_u.to(device), ang_u.to(device)
        pt_s, ang_s = pt_s.to(device), ang_s.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            t_logits = teacher((pt_u, ang_u))

        opt.zero_grad()
        s_logits = student((pt_s, ang_s))

        hard = bce_logits(s_logits.view(-1), labels.view(-1))
        kd = kd_loss_binary(s_logits, t_logits, T)
        loss = alpha_kd * kd + (1.0 - alpha_kd) * hard

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        acc = batch_accuracy_from_logits(s_logits, labels)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


# ----------------------------- Main ----------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train top tagging model with EFN KD")
    parser.add_argument(
        "--train_path",
        type=str,
        default="/home/ryan/ComputerScience/ATLAS/ATLAS-top-tagging-open-data/data",
        help="Path to directory containing training data files",
    )
    args = parser.parse_args()

    # Paths and data selection
    train_path = Path(args.train_path)
    train_files = sorted(list(train_path.glob("*.h5")))

    n_train_jets = 30000
    valid_fraction = 0.176
    max_constits = 40

    tagger_type = "efn"
    num_epochs = 30
    batch_size = 64

    # KD hyperparams (same spirit as your transformer notebook)
    TEMPERATURE = 3.0
    ALPHA_KD = 0.5

    figure_dir = Path().cwd() / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Read data and prepare for tagger training")

    # utils.load_from_files returns weights, but we ignore them here
    all_data, all_labels, _, _, all_pt = utils.load_from_files(
        train_files,
        max_jets=n_train_jets,
        max_constits=max_constits,
        use_train_weights=False
    )

    print("Applying Gaussian smearing to data (10% eta/phi smear)")
    smeared_data = copy.deepcopy(all_data)
    smeared_data = smear_dataset(smeared_data, eta_smear_factor=0.15, phi_smear_factor=0.10)

    # Split off test
    test_fraction = 0.15
    test_size = int(len(all_data) * test_fraction)
    test_idx = len(all_data) - test_size

    print(f"Data split: {test_idx} for train+val (will be split 70/15), {test_size} for test")

    # Save test data for later evaluation (both unsmeared and smeared)
    test_data_dir = Path().cwd() / "test_split"
    test_data_dir.mkdir(exist_ok=True)

    # Save unsmeared test data
    np.savez(
        test_data_dir / "test_data_unsmeared.npz",
        data=all_data[test_idx:],
        labels=all_labels[test_idx:],
        pt=all_pt[test_idx:]
    )
    print(f"Saved {test_size} unsmeared test jets to {test_data_dir / 'test_data_unsmeared.npz'}")

    # Save smeared test data
    np.savez(
        test_data_dir / "test_data_smeared.npz",
        data=smeared_data[test_idx:],
        labels=all_labels[test_idx:],
        pt=all_pt[test_idx:]
    )
    print(f"Saved {test_size} smeared test jets to {test_data_dir / 'test_data_smeared.npz'}")

    # Keep train+val slices
    train_data = all_data[:test_idx].copy()
    train_labels = all_labels[:test_idx].copy()
    smeared_train_data = smeared_data[:test_idx].copy()

    # Free memory
    del all_data, all_labels, all_pt, smeared_data

    print(f"Remaining {len(train_data)} jets will be split into train/validation by train_test_split")

    if tagger_type != "efn":
        raise ValueError("This drop-in file implements EFN KD only (tagger_type must be 'efn').")

    # Prepare unsmeared (teacher input)
    train_angular = train_data[:, :, 0:2]
    train_pt = train_data[:, :, 2]

    # Prepare smeared (student input)
    smeared_train_angular = smeared_train_data[:, :, 0:2]
    smeared_train_pt = smeared_train_data[:, :, 2]

    # Split both unsmeared and smeared with same indices
    (
        train_angular, valid_angular,
        train_pt, valid_pt,
        smeared_train_angular, smeared_valid_angular,
        smeared_train_pt, smeared_valid_pt,
        train_labels, valid_labels
    ) = train_test_split(
        train_angular,
        train_pt,
        smeared_train_angular,
        smeared_train_pt,
        train_labels,
        test_size=valid_fraction,
        random_state=RANDOM_SEED,
        stratify=train_labels
    )

    # Paired datasets
    train_ds = PairedEFN_Dataset(
        train_pt, train_angular,
        smeared_train_pt, smeared_train_angular,
        train_labels
    )
    valid_ds = PairedEFN_Dataset(
        valid_pt, valid_angular,
        smeared_valid_pt, smeared_valid_angular,
        valid_labels
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # Models
    efn_unsmeared = EFN()
    efn_smeared = EFN()
    efn_untaught = EFN()  # Third model: untaught student

    opt_unsmeared = torch.optim.Adam(efn_unsmeared.parameters(), lr=6.3e-5)
    opt_smeared = torch.optim.Adam(efn_smeared.parameters(), lr=6.3e-5)
    opt_untaught = torch.optim.Adam(efn_untaught.parameters(), lr=6.3e-5)

    device = torch.device("cpu")  # Force CPU to avoid GPU driver issues
    efn_unsmeared = efn_unsmeared.to(device)
    efn_smeared = efn_smeared.to(device)
    efn_untaught = efn_untaught.to(device)

    # Checkpoints (kept same directory style and filenames)
    checkpoint_dir = Path().cwd() / "checkpoints" / "efn_teaching"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss_unsmeared = float("inf")
    best_val_loss_smeared = float("inf")
    best_val_loss_untaught = float("inf")

    teacher_ckpt_path = checkpoint_dir / "best_model_unsmeared.pt"
    student_ckpt_path = checkpoint_dir / "best_model_smeared.pt"
    untaught_ckpt_path = checkpoint_dir / "best_model_untaught.pt"

    # -------------------- Stage A: Train TEACHER (unsmeared) -------------------- #
    print("\n====================")
    print("STAGE A: Training UNSMEARED teacher (hard labels only)")
    print("====================")

    for epoch in range(num_epochs):
        train_loss_u, train_acc_u = train_teacher_epoch(efn_unsmeared, train_loader, opt_unsmeared, device)
        val_loss_u, val_acc_u = eval_teacher(efn_unsmeared, valid_loader, device)

        print(
            f"UNSMEARED Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss_u:.4f}, Train Acc: {train_acc_u:.4f} | "
            f"Val Loss: {val_loss_u:.4f}, Val Acc: {val_acc_u:.4f}"
        )

        if val_loss_u < best_val_loss_unsmeared:
            best_val_loss_unsmeared = val_loss_u
            torch.save(efn_unsmeared.state_dict(), teacher_ckpt_path)
            print(f"UNSMEARED  → Saved checkpoint: {teacher_ckpt_path} (val_loss: {val_loss_u:.4f})")

    # Load best teacher and freeze
    efn_unsmeared.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
    efn_unsmeared.eval()
    for p in efn_unsmeared.parameters():
        p.requires_grad = False

    # -------------------- Stage B: Train STUDENT (smeared) with KD -------------------- #
    print("\n====================")
    print("STAGE B: Training SMEARED student with KD from teacher")
    print(f"KD settings: T={TEMPERATURE}, alpha_kd={ALPHA_KD} (hard weight={1.0-ALPHA_KD})")
    print("====================")

    for epoch in range(num_epochs):
        train_loss_s, train_acc_s = train_student_kd_epoch(
            efn_smeared, efn_unsmeared, train_loader, opt_smeared, device,
            T=TEMPERATURE, alpha_kd=ALPHA_KD
        )
        val_loss_s, val_acc_s = eval_student_kd(
            efn_smeared, efn_unsmeared, valid_loader, device,
            T=TEMPERATURE, alpha_kd=ALPHA_KD
        )

        print(
            f"SMEARED Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss_s:.4f}, Train Acc: {train_acc_s:.4f} | "
            f"Val Loss: {val_loss_s:.4f}, Val Acc: {val_acc_s:.4f}"
        )

        if val_loss_s < best_val_loss_smeared:
            best_val_loss_smeared = val_loss_s
            torch.save(efn_smeared.state_dict(), student_ckpt_path)
            print(f"SMEARED  → Saved checkpoint: {student_ckpt_path} (val_loss: {val_loss_s:.4f})")

    # -------------------- Stage C: Train UNTAUGHT STUDENT (smeared, no KD) -------------------- #
    print("\n====================")
    print("STAGE C: Training UNTAUGHT student (smeared, no teacher)")
    print("====================")

    for epoch in range(num_epochs):
        train_loss_u, train_acc_u = train_student_independent_epoch(
            efn_untaught, train_loader, opt_untaught, device
        )
        val_loss_u, val_acc_u = eval_student_independent(
            efn_untaught, valid_loader, device
        )

        print(
            f"UNTAUGHT Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss_u:.4f}, Train Acc: {train_acc_u:.4f} | "
            f"Val Loss: {val_loss_u:.4f}, Val Acc: {val_acc_u:.4f}"
        )

        if val_loss_u < best_val_loss_untaught:
            best_val_loss_untaught = val_loss_u
            torch.save(efn_untaught.state_dict(), untaught_ckpt_path)
            print(f"UNTAUGHT  → Saved checkpoint: {untaught_ckpt_path} (val_loss: {val_loss_u:.4f})")


if __name__ == "__main__":
    main()
