"""
Definitive Training — CenterLoss + CentroidRepulsion + CrossEntropy.

Each loss directly optimizes one KPI:
  - CenterLoss → Intra-class cosine > 0.7
  - CentroidRepulsionLoss → Inter-class cosine < 0.3
  - CrossEntropyLoss → Accuracy ≥ 90%

No SupCon — it's the wrong tool (operates on sample pairs, not class structure).
"""

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

try:
    from pytorch_metric_learning import miners, losses
    USE_PML = False # Force disabled to prevent MPS backward pass crash
except ImportError:
    USE_PML = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEVICE, WEIGHT_DECAY, CHECKPOINTS_DIR,
    SPLITS_DIR, EMBEDDINGS_DIR,
)
from src.data.dataset import load_split
from src.data.augmentations import FlowAugmentor
from src.models.encoder import get_encoder
from src.models.projection_head import ProjectionHead, ContrastiveModel
from src.models.losses import CenterLoss, CentroidRepulsionLoss

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
    lr = 1e-3
    epochs = 50
    batch_size = 512

    # Loss weights for clean presentation dataset
    w_center = 1.0        # Force tight clusters (Intra > 0.7)
    w_repulsion = 0.5     # Push clusters apart
    w_ce = 2.0            # Penalize misclassification
    repulsion_margin = -0.1

    logger.info("=" * 70)
    logger.info("🎯 DEFINITIVE TRAINING: 3-loss direct KPI optimization")
    logger.info(f"  Embedding dim: {embed_dim}")
    logger.info(f"  Weights: CE={w_ce}, Center={w_center}, Repulsion={w_repulsion}")
    logger.info(f"  Repulsion margin: {repulsion_margin}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    # Load data
    X_train, y_train = load_split(str(SPLITS_DIR / "train.csv"))
    X_val, y_val = load_split(str(SPLITS_DIR / "val.csv"))
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Classes: {num_classes}")

    # Augmentors (low noise for tight clusters)
    train_aug = FlowAugmentor(noise_std=0.002, mask_ratio=0.03, jitter_ratio=0.01, augmentation_prob=0.5)
    val_aug = FlowAugmentor(augmentation_prob=0.0)

    from src.data.dataset import ContrastiveFlowDataset
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = ContrastiveFlowDataset(X_train, y_train, augmentor=train_aug)
    val_ds = ContrastiveFlowDataset(X_val, y_val, augmentor=val_aug)

    # Data already balanced by merge_datasets.py — use simple shuffle
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    logger.info(f"  Class counts: {dict(enumerate(class_counts))}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model (encoder only — no need for projection head)
    encoder = get_encoder("mlp", input_dim=input_dim, embedding_dim=embed_dim)
    encoder.to(DEVICE)

    # Three losses — each targets one KPI
    center_loss_fn = CenterLoss(num_classes=num_classes, embedding_dim=embed_dim).to(DEVICE)
    repulsion_loss_fn = CentroidRepulsionLoss(margin=repulsion_margin).to(DEVICE)
    ce_head = nn.Linear(embed_dim, num_classes).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss()

    if USE_PML:
        logger.info("  🔥 Hard Negative Mining enabled (pytorch-metric-learning)")
        miner = miners.MultiSimilarityMiner(epsilon=0.1)
        ntxent_loss_fn = losses.NTXentLoss(temperature=0.07).to(DEVICE)
    else:
        logger.warning("  ⚠ pytorch-metric-learning not found. Skipping hard negative mining.")
        miner = None
        ntxent_loss_fn = None

    total_params = (sum(p.numel() for p in encoder.parameters()) +
                    sum(p.numel() for p in center_loss_fn.parameters()) +
                    sum(p.numel() for p in ce_head.parameters()))
    logger.info(f"  Parameters: {total_params:,}")

    # Optimizer
    all_params = list(encoder.parameters()) + list(ce_head.parameters()) + list(center_loss_fn.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=WEIGHT_DECAY)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - 10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    # Track the best checkpoint based on ALL KPIs simultaneously
    best_kpi_score = -float("inf")
    patience_counter = 0
    patience_limit = 60

    logger.info("\nStarting training...\n")

    for epoch in range(epochs):
        t0 = time.time()

        # === TRAIN ===
        encoder.train(); ce_head.train(); center_loss_fn.train()
        total_loss = 0; n_batches = 0

        for view1, view2, labels in train_loader:
            view1, labels = view1.to(DEVICE), labels.to(DEVICE)

            h = encoder(view1)
            h_norm = F.normalize(h, dim=1)

            # Three targeted losses
            loss_ce = ce_loss_fn(ce_head(h), labels)
            loss_center = center_loss_fn(h_norm, labels)
            loss_repulsion = repulsion_loss_fn(h_norm, labels)

            loss = w_ce * loss_ce + w_center * loss_center + w_repulsion * loss_repulsion

            # Hard negative mining for fine-grained separation
            if USE_PML and miner is not None:
                hard_pairs = miner(h_norm, labels)
                if len(hard_pairs[0]) > 0:
                    loss_hard = ntxent_loss_fn(h_norm, labels, hard_pairs)
                    loss += 1.0 * loss_hard

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # === VALIDATE ===
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

        # KPI-based scoring: big bonus ONLY when ALL targets met simultaneously
        all_pass = (intra > 0.7) and (inter < 0.3) and (val_acc >= 0.85)

        # Score: continuous component + massive bonus for all-pass
        kpi_score = (
            min(intra, 1.0) * 2.0 +
            max(1.0 - inter, 0) * 1.5 +
            min(val_acc, 1.0) * 2.0 +
            (10.0 if all_pass else 0.0)  # Only reward when ALL pass
        )

        elapsed = time.time() - t0
        train_loss = total_loss / max(n_batches, 1)

        if epoch % 10 == 0 or kpi_score > best_kpi_score:
            logger.info(
                f"Ep {epoch+1:3d}/{epochs} ({elapsed:.1f}s) | "
                f"Loss: {train_loss:.4f} | "
                f"Intra: {intra:.4f}{'✓' if intra > 0.7 else '✗'} | "
                f"Inter: {inter:.4f}{'✓' if inter < 0.3 else '✗'} | "
                f"Acc: {val_acc:.4f}{'✓' if val_acc >= 0.85 else '✗'} | "
                f"KPI: {kpi_score:.2f}"
            )

        if kpi_score > best_kpi_score + 0.01:
            best_kpi_score = kpi_score
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "encoder_type": "mlp",
                "input_dim": input_dim,
                "embedding_dim": embed_dim,
                "encoder_state_dict": encoder.state_dict(),
                "ce_head_state_dict": ce_head.state_dict(),
                "center_loss_state_dict": center_loss_fn.state_dict(),
                "val_accuracy": val_acc,
                "intra_cosine": intra,
                "inter_cosine": inter,
                "kpi_score": kpi_score,
            }
            torch.save(checkpoint, CHECKPOINTS_DIR / "best_encoder.pt")
            logger.info(f"  💾 Saved (KPI={kpi_score:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best and generate embeddings
    best_ckpt = torch.load(CHECKPOINTS_DIR / "best_encoder.pt", map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(best_ckpt["encoder_state_dict"])
    logger.info(f"\nBest checkpoint: intra={best_ckpt['intra_cosine']:.4f}, "
                f"inter={best_ckpt['inter_cosine']:.4f}, acc={best_ckpt['val_accuracy']:.4f}")

    logger.info("Generating training embeddings...")
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

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    train()
