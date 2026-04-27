"""
Enhanced Training — Combined SupCon + CenterLoss + CrossEntropy.

Achieves both high intra-class cosine (compactness) AND accuracy by:
1. SupCon: pushes different-class embeddings apart  
2. CenterLoss: pulls same-class embeddings toward class centroid
3. CrossEntropy: adds direct classification gradient signal

Usage:
    python scripts/train_enhanced.py
    python scripts/train_enhanced.py --embedding-dim 64 --epochs 200
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
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    WARMUP_EPOCHS, TEMPERATURE, EMBEDDING_DIM, CHECKPOINTS_DIR,
    SPLITS_DIR, EMBEDDINGS_DIR, PATIENCE, MIN_DELTA,
)
from src.data.dataset import load_split, get_dataloaders
from src.data.augmentations import FlowAugmentor
from src.models.encoder import get_encoder
from src.models.projection_head import ProjectionHead, ContrastiveModel
from src.models.losses import SupConLoss, CenterLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced training with combined loss")
    parser.add_argument("--encoder", type=str, default="mlp", choices=["transformer", "cnn_bilstm", "mlp"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--center-weight", type=float, default=0.5,
                        help="Weight for center loss (compactness)")
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="Weight for cross-entropy loss (accuracy)")
    parser.add_argument("--supcon-weight", type=float, default=1.0,
                        help="Weight for SupCon loss (separation)")
    parser.add_argument("--aug-noise", type=float, default=0.005,
                        help="Augmentation noise (lower = tighter clusters)")
    parser.add_argument("--aug-mask", type=float, default=0.1,
                        help="Augmentation mask ratio")
    parser.add_argument("--train-data", type=str, default=str(SPLITS_DIR / "train.csv"))
    parser.add_argument("--val-data", type=str, default=str(SPLITS_DIR / "val.csv"))
    return parser.parse_args()


def train(args):
    logger.info("🏋️ Enhanced Training Configuration:")
    logger.info(f"  Encoder: {args.encoder}")
    logger.info(f"  Embedding dim: {args.embedding_dim}")
    logger.info(f"  Loss weights: SupCon={args.supcon_weight}, Center={args.center_weight}, CE={args.ce_weight}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Aug noise: {args.aug_noise}, mask: {args.aug_mask}")
    logger.info(f"  Device: {DEVICE}")

    # Load data
    X_train, y_train = load_split(args.train_data)
    X_val, y_val = load_split(args.val_data)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    logger.info(f"  Input dim: {input_dim}, Train: {len(X_train)}, Val: {len(X_val)}, Classes: {num_classes}")

    # Custom augmentor with reduced noise for tighter clusters
    train_augmentor = FlowAugmentor(
        noise_std=args.aug_noise,
        mask_ratio=args.aug_mask,
        jitter_ratio=0.02,
        augmentation_prob=0.7,
    )
    val_augmentor = FlowAugmentor(augmentation_prob=0.0)

    from src.data.dataset import ContrastiveFlowDataset
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_dataset = ContrastiveFlowDataset(X_train, y_train, augmentor=train_augmentor)
    val_dataset = ContrastiveFlowDataset(X_val, y_val, augmentor=val_augmentor)

    # Class-balanced sampling — ensures minority classes (voip, xr_ar) get
    # equal representation in each batch, critical for balanced embeddings
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    logger.info(f"  Class distribution: {dict(zip(range(num_classes), class_counts))}")
    logger.info(f"  Class weights: {dict(zip(range(num_classes), np.round(class_weights / class_weights.min(), 2)))}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    encoder = get_encoder(args.encoder, input_dim=input_dim, embedding_dim=args.embedding_dim)
    projection_head = ProjectionHead(input_dim=args.embedding_dim, hidden_dim=128, output_dim=128)
    model = ContrastiveModel(encoder, projection_head).to(DEVICE)

    # Losses — class-weighted CE to handle imbalance
    supcon_loss_fn = SupConLoss(temperature=args.temperature)
    center_loss_fn = CenterLoss(num_classes=num_classes, embedding_dim=args.embedding_dim).to(DEVICE)
    ce_head = nn.Linear(args.embedding_dim, num_classes).to(DEVICE)
    ce_class_weights = torch.FloatTensor(class_weights / class_weights.min()).to(DEVICE)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_class_weights)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in center_loss_fn.parameters()) + sum(p.numel() for p in ce_head.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # Optimizer — include center loss params and CE head
    all_params = list(model.parameters()) + list(center_loss_fn.parameters()) + list(ce_head.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=WEIGHT_DECAY)

    # Scheduler
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    # Training loop — save based on combined KPI score, not just val_loss
    best_kpi_score = -float("inf")
    patience_counter = 0
    patience_limit = 25  # More patience for complex multi-objective optimization
    history = {"train_loss": [], "val_loss": [], "intra_cosine": [], "inter_cosine": [], "accuracy": []}

    logger.info("=" * 70)
    logger.info("Starting enhanced training...")
    logger.info("=" * 70)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # === TRAIN ===
        model.train()
        center_loss_fn.train()
        ce_head.train()
        total_loss = 0
        num_batches = 0

        for view1, view2, labels in train_loader:
            view1, view2, labels = view1.to(DEVICE), view2.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            z1 = model(view1)           # projected embeddings
            z2 = model(view2)
            h1 = model.encoder(view1)   # raw encoder embeddings

            # Combined loss
            loss_sc = supcon_loss_fn(z1, z2, labels)
            h1_norm = F.normalize(h1, dim=1)
            loss_center = center_loss_fn(h1_norm, labels)
            logits = ce_head(h1)
            loss_ce = ce_loss_fn(logits, labels)

            loss = (args.supcon_weight * loss_sc +
                    args.center_weight * loss_center +
                    args.ce_weight * loss_ce)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        # === VALIDATE ===
        model.eval()
        val_total_loss = 0
        val_batches = 0
        all_embeddings = []
        all_labels = []
        correct = 0
        total = 0

        with torch.no_grad():
            for view1, view2, labels in val_loader:
                view1, view2, labels = view1.to(DEVICE), view2.to(DEVICE), labels.to(DEVICE)

                z1 = model(view1)
                z2 = model(view2)
                h1 = model.encoder(view1)

                loss_sc = supcon_loss_fn(z1, z2, labels)
                h1_norm = F.normalize(h1, dim=1)
                loss_center = center_loss_fn(h1_norm, labels)
                logits = ce_head(h1)
                loss_ce = ce_loss_fn(logits, labels)

                loss = (args.supcon_weight * loss_sc +
                        args.center_weight * loss_center +
                        args.ce_weight * loss_ce)

                val_total_loss += loss.item()
                val_batches += 1

                all_embeddings.append(h1_norm.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # CE accuracy
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        val_loss = val_total_loss / max(val_batches, 1)
        val_acc = correct / max(total, 1)

        # Compute cosine metrics
        embeddings = np.concatenate(all_embeddings)
        labels_np = np.concatenate(all_labels)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
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
            for lb in unique_labels[i+1:]:
                cross = sim_matrix[np.ix_(labels_np == la, labels_np == lb)]
                inter_scores.append(cross.mean())

        intra_avg = np.mean(intra_scores) if intra_scores else 0.0
        inter_avg = np.mean(inter_scores) if inter_scores else 1.0

        elapsed = time.time() - epoch_start

        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Intra: {intra_avg:.4f} | Inter: {inter_avg:.4f} | "
            f"CE-Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["intra_cosine"].append(intra_avg)
        history["inter_cosine"].append(inter_avg)
        history["accuracy"].append(val_acc)

        # Combined KPI score: rewards intra>0.7, inter<0.3, acc>0.9
        # This selects the checkpoint that best balances ALL metrics
        kpi_score = (
            min(intra_avg, 1.0) * 2.0 +          # Reward high intra (weight 2x)
            max(1.0 - inter_avg, 0.0) * 1.5 +    # Reward low inter (weight 1.5x)
            min(val_acc, 1.0) * 2.0               # Reward high accuracy (weight 2x)
        )

        if kpi_score > best_kpi_score + 0.001:
            best_kpi_score = kpi_score
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "encoder_type": args.encoder,
                "input_dim": input_dim,
                "embedding_dim": args.embedding_dim,
                "encoder_state_dict": encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "ce_head_state_dict": ce_head.state_dict(),
                "center_loss_state_dict": center_loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "intra_cosine": intra_avg,
                "inter_cosine": inter_avg,
                "val_accuracy": val_acc,
                "kpi_score": kpi_score,
                "history": history,
            }
            torch.save(checkpoint, CHECKPOINTS_DIR / "best_encoder.pt")
            logger.info(f"  💾 Saved (KPI={kpi_score:.3f}, intra={intra_avg:.4f}, inter={inter_avg:.4f}, acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save final
    torch.save({
        "epoch": epoch,
        "encoder_type": args.encoder,
        "input_dim": input_dim,
        "embedding_dim": args.embedding_dim,
        "encoder_state_dict": encoder.state_dict(),
        "model_state_dict": model.state_dict(),
        "history": history,
    }, CHECKPOINTS_DIR / "final_encoder.pt")

    # Generate embeddings for FAISS
    logger.info("Generating training embeddings...")
    model.eval()
    train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(train_tensor), 1024):
            batch = train_tensor[i:i+1024]
            h = model.encoder(batch)
            h = F.normalize(h, dim=1)
            all_emb.append(h.cpu().numpy())

    train_embeddings = np.concatenate(all_emb)
    np.save(EMBEDDINGS_DIR / "train_embeddings.npy", train_embeddings)
    np.save(EMBEDDINGS_DIR / "train_labels.npy", y_train)

    # Build FAISS index
    from src.classifier.knn_classifier import FAISSKNNClassifier
    knn = FAISSKNNClassifier(embedding_dim=args.embedding_dim)
    knn.fit(train_embeddings, y_train)
    knn.save()

    logger.info(f"✅ Training complete! Best KPI score: {best_kpi_score:.4f}")
    logger.info(f"   Best intra cosine: {max(history['intra_cosine']):.4f}")
    logger.info(f"   Best val accuracy: {max(history['accuracy']):.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
