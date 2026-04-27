"""
Two-Phase Training — Phase 1: Compact clusters, Phase 2: Push them apart.

Phase 1 (CE + CenterLoss): Builds tight, accurate clusters
Phase 2 (+ SupCon): Separates clusters while preserving compactness

This avoids the SupCon vs CenterLoss conflict that plagued single-phase training.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEVICE, WEIGHT_DECAY, CHECKPOINTS_DIR,
    SPLITS_DIR, EMBEDDINGS_DIR,
)
from src.data.dataset import load_split
from src.data.augmentations import FlowAugmentor
from src.models.encoder import get_encoder
from src.models.projection_head import ProjectionHead, ContrastiveModel
from src.models.losses import SupConLoss, CenterLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(embeddings_np, labels_np):
    """Compute intra/inter class cosine similarities."""
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    normalized = embeddings_np / (norms + 1e-8)
    sim_matrix = normalized @ normalized.T

    intra_scores = []
    for label in np.unique(labels_np):
        mask = labels_np == label
        if mask.sum() < 2:
            continue
        class_sim = sim_matrix[np.ix_(mask, mask)]
        n = class_sim.shape[0]
        off_diag = class_sim[~np.eye(n, dtype=bool)]
        intra_scores.append(off_diag.mean())

    inter_scores = []
    unique_labels = np.unique(labels_np)
    for i, la in enumerate(unique_labels):
        for lb in unique_labels[i + 1:]:
            cross = sim_matrix[np.ix_(labels_np == la, labels_np == lb)]
            inter_scores.append(cross.mean())

    return (
        np.mean(intra_scores) if intra_scores else 0.0,
        np.mean(inter_scores) if inter_scores else 1.0,
    )


def train():
    embed_dim = 48
    lr = 5e-4
    phase1_epochs = 150
    phase2_epochs = 300
    batch_size = 128
    temperature = 0.25

    logger.info("=" * 70)
    logger.info("🏋️ TWO-PHASE TRAINING")
    logger.info(f"  Embedding dim: {embed_dim}")
    logger.info(f"  Phase 1: {phase1_epochs} epochs (CE + CenterLoss)")
    logger.info(f"  Phase 2: {phase2_epochs} epochs (CE + CenterLoss + SupCon)")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    # Load data
    X_train, y_train = load_split(str(SPLITS_DIR / "train.csv"))
    X_val, y_val = load_split(str(SPLITS_DIR / "val.csv"))
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Classes: {num_classes}")

    # Augmentors
    train_aug = FlowAugmentor(noise_std=0.002, mask_ratio=0.03, jitter_ratio=0.01, augmentation_prob=0.6)
    val_aug = FlowAugmentor(augmentation_prob=0.0)

    from src.data.dataset import ContrastiveFlowDataset
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = ContrastiveFlowDataset(X_train, y_train, augmentor=train_aug)
    val_ds = ContrastiveFlowDataset(X_val, y_val, augmentor=val_aug)

    # Class-balanced sampling
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    logger.info(f"  Class counts: {dict(enumerate(class_counts))}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    encoder = get_encoder("mlp", input_dim=input_dim, embedding_dim=embed_dim)
    proj_head = ProjectionHead(input_dim=embed_dim, hidden_dim=128, output_dim=128)
    model = ContrastiveModel(encoder, proj_head).to(DEVICE)

    # Loss components
    center_loss_fn = CenterLoss(num_classes=num_classes, embedding_dim=embed_dim).to(DEVICE)
    ce_head = nn.Linear(embed_dim, num_classes).to(DEVICE)
    ce_class_w = torch.FloatTensor(class_weights / class_weights.min()).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_class_w)
    supcon_loss_fn = SupConLoss(temperature=temperature)

    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ═══════════════════════════════════════════════
    # PHASE 1: Build compact, accurate clusters
    # ═══════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 1: Building compact clusters (CE + CenterLoss)")
    logger.info("═" * 70)

    all_params_p1 = list(encoder.parameters()) + list(ce_head.parameters()) + list(center_loss_fn.parameters())
    opt1 = optim.AdamW(all_params_p1, lr=lr, weight_decay=WEIGHT_DECAY)
    warmup1 = LinearLR(opt1, start_factor=0.01, total_iters=10)
    cosine1 = CosineAnnealingLR(opt1, T_max=phase1_epochs - 10, eta_min=1e-6)
    sched1 = SequentialLR(opt1, [warmup1, cosine1], milestones=[10])

    best_kpi = -float("inf")
    patience = 0

    for epoch in range(phase1_epochs):
        t0 = time.time()

        # Train
        encoder.train(); ce_head.train(); center_loss_fn.train()
        total_loss = 0; n_batches = 0

        for view1, view2, labels in train_loader:
            view1, labels = view1.to(DEVICE), labels.to(DEVICE)
            h = encoder(view1)
            h_norm = F.normalize(h, dim=1)

            loss_ce = ce_loss_fn(ce_head(h), labels)
            loss_center = center_loss_fn(h_norm, labels)
            loss = 2.0 * loss_ce + 0.1 * loss_center

            opt1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params_p1, 1.0)
            opt1.step()
            total_loss += loss.item(); n_batches += 1

        sched1.step()

        # Validate
        encoder.eval(); ce_head.eval()
        all_emb, all_lbl = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for v1, v2, lbl in val_loader:
                v1, lbl = v1.to(DEVICE), lbl.to(DEVICE)
                h = encoder(v1)
                h_norm = F.normalize(h, dim=1)
                logits = ce_head(h)
                preds = logits.argmax(1)
                correct += (preds == lbl).sum().item()
                total += len(lbl)
                all_emb.append(h_norm.cpu().numpy())
                all_lbl.append(lbl.cpu().numpy())

        val_acc = correct / total
        embeddings = np.concatenate(all_emb)
        labels_np = np.concatenate(all_lbl)
        intra, inter = compute_metrics(embeddings, labels_np)

        kpi = min(intra, 1.0) * 2 + max(1 - inter, 0) * 1.5 + min(val_acc, 1.0) * 2
        elapsed = time.time() - t0

        if epoch % 5 == 0 or kpi > best_kpi:
            logger.info(
                f"P1 Ep {epoch+1}/{phase1_epochs} ({elapsed:.1f}s) | "
                f"Intra: {intra:.4f} | Inter: {inter:.4f} | "
                f"Acc: {val_acc:.4f} | KPI: {kpi:.3f}"
            )

        if kpi > best_kpi + 0.001:
            best_kpi = kpi
            patience = 0
            # Save Phase 1 best
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "ce_head_state_dict": ce_head.state_dict(),
                "center_loss_state_dict": center_loss_fn.state_dict(),
            }, CHECKPOINTS_DIR / "phase1_best.pt")
        else:
            patience += 1
            if patience >= 30:
                logger.info(f"P1 early stop at epoch {epoch+1}")
                break

    # Load Phase 1 best for Phase 2
    p1_ckpt = torch.load(CHECKPOINTS_DIR / "phase1_best.pt", map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(p1_ckpt["encoder_state_dict"])
    ce_head.load_state_dict(p1_ckpt["ce_head_state_dict"])
    center_loss_fn.load_state_dict(p1_ckpt["center_loss_state_dict"])

    logger.info(f"\nPhase 1 done. Best KPI: {best_kpi:.4f}")

    # ═══════════════════════════════════════════════
    # PHASE 2: Push clusters apart with SupCon
    # ═══════════════════════════════════════════════
    logger.info("\n" + "═" * 70)
    logger.info("PHASE 2: Separating clusters (+SupCon)")
    logger.info("═" * 70)

    all_params_p2 = (list(model.parameters()) + list(ce_head.parameters()) +
                     list(center_loss_fn.parameters()))
    # Higher LR to allow meaningful separation
    opt2 = optim.AdamW(all_params_p2, lr=lr, weight_decay=WEIGHT_DECAY)
    cosine2 = CosineAnnealingLR(opt2, T_max=phase2_epochs, eta_min=1e-6)

    best_kpi2 = -float("inf")
    patience2 = 0

    for epoch in range(phase2_epochs):
        t0 = time.time()

        model.train(); ce_head.train(); center_loss_fn.train()
        total_loss = 0; n_batches = 0

        for view1, view2, labels in train_loader:
            view1, view2, labels = view1.to(DEVICE), view2.to(DEVICE), labels.to(DEVICE)

            # Contrastive views through full model (encoder + projection)
            z1 = model(view1)
            z2 = model(view2)

            # Encoder embeddings for CE + center
            h1 = model.encoder(view1)
            h1_norm = F.normalize(h1, dim=1)

            loss_supcon = supcon_loss_fn(z1, z2, labels)
            loss_ce = ce_loss_fn(ce_head(h1), labels)

            # SupCon ONLY + CE for accuracy — NO center loss (it fights SupCon)
            loss = 15.0 * loss_supcon + 2.0 * loss_ce

            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params_p2, 1.0)
            opt2.step()
            total_loss += loss.item(); n_batches += 1

        cosine2.step()

        # Validate
        model.eval(); ce_head.eval()
        all_emb, all_lbl = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for v1, v2, lbl in val_loader:
                v1, lbl = v1.to(DEVICE), lbl.to(DEVICE)
                h = model.encoder(v1)
                h_norm = F.normalize(h, dim=1)
                logits = ce_head(h)
                preds = logits.argmax(1)
                correct += (preds == lbl).sum().item()
                total += len(lbl)
                all_emb.append(h_norm.cpu().numpy())
                all_lbl.append(lbl.cpu().numpy())

        val_acc = correct / total
        embeddings = np.concatenate(all_emb)
        labels_np = np.concatenate(all_lbl)
        intra, inter = compute_metrics(embeddings, labels_np)

        kpi = min(intra, 1.0) * 2 + max(1 - inter, 0) * 1.5 + min(val_acc, 1.0) * 2
        elapsed = time.time() - t0

        if epoch % 5 == 0 or kpi > best_kpi2:
            logger.info(
                f"P2 Ep {epoch+1}/{phase2_epochs} ({elapsed:.1f}s) | "
                f"Intra: {intra:.4f} | Inter: {inter:.4f} | "
                f"Acc: {val_acc:.4f} | KPI: {kpi:.3f}"
            )

        if kpi > best_kpi2 + 0.001:
            best_kpi2 = kpi
            patience2 = 0
            checkpoint = {
                "epoch": epoch,
                "encoder_type": "mlp",
                "input_dim": input_dim,
                "embedding_dim": embed_dim,
                "encoder_state_dict": encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "ce_head_state_dict": ce_head.state_dict(),
                "center_loss_state_dict": center_loss_fn.state_dict(),
                "optimizer_state_dict": opt2.state_dict(),
                "val_accuracy": val_acc,
                "intra_cosine": intra,
                "inter_cosine": inter,
                "kpi_score": kpi,
            }
            torch.save(checkpoint, CHECKPOINTS_DIR / "best_encoder.pt")
            logger.info(f"  💾 Saved")
        else:
            patience2 += 1
            if patience2 >= 40:
                logger.info(f"P2 early stop at epoch {epoch+1}")
                break

    # Generate embeddings for k-NN
    logger.info("\nGenerating training embeddings...")
    encoder.eval()
    train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(train_tensor), 1024):
            h = encoder(train_tensor[i:i+1024])
            h = F.normalize(h, dim=1)
            all_emb.append(h.cpu().numpy())

    train_embeddings = np.concatenate(all_emb)
    np.save(EMBEDDINGS_DIR / "train_embeddings.npy", train_embeddings)
    np.save(EMBEDDINGS_DIR / "train_labels.npy", y_train)

    from src.classifier.knn_classifier import FAISSKNNClassifier
    knn = FAISSKNNClassifier(embedding_dim=embed_dim)
    knn.fit(train_embeddings, y_train)
    knn.save()

    logger.info(f"✅ Done! Phase 1 KPI: {best_kpi:.3f}, Phase 2 KPI: {best_kpi2:.3f}")


if __name__ == "__main__":
    train()
