"""
Training Script — Full contrastive learning training pipeline.

Trains the Transformer encoder with NT-Xent loss on flow embeddings.
Supports multiple loss functions, learning rate scheduling, early stopping,
and checkpoint saving.

Usage:
    python scripts/train.py --encoder transformer --loss ntxent --epochs 100
    python scripts/train.py --encoder cnn_bilstm --loss supcon --epochs 50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    WARMUP_EPOCHS, TEMPERATURE, EMBEDDING_DIM, CHECKPOINTS_DIR,
    SPLITS_DIR, EMBEDDINGS_DIR, PATIENCE, MIN_DELTA,
)
from src.data.dataset import load_split, get_dataloaders
from src.models.encoder import get_encoder
from src.models.projection_head import ProjectionHead, ContrastiveModel
from src.models.losses import NTXentLoss, SupConLoss, TripletLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train flow encoder with contrastive learning")
    parser.add_argument("--encoder", type=str, default="mlp", choices=["transformer", "cnn_bilstm", "mlp"])
    parser.add_argument("--loss", type=str, default="ntxent", choices=["ntxent", "supcon", "triplet"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--train-data", type=str, default=str(SPLITS_DIR / "train.csv"))
    parser.add_argument("--val-data", type=str, default=str(SPLITS_DIR / "val.csv"))
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def create_loss_fn(loss_type: str, temperature: float):
    """Create the appropriate loss function."""
    if loss_type == "ntxent":
        return NTXentLoss(temperature=temperature)
    elif loss_type == "supcon":
        return SupConLoss(temperature=temperature)
    elif loss_type == "triplet":
        return TripletLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(
    model: ContrastiveModel,
    train_loader,
    loss_fn,
    optimizer,
    loss_type: str,
    device: str,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        if loss_type in ("ntxent", "supcon"):
            view1, view2, labels = batch
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)

            z1 = model(view1)
            z2 = model(view2)

            if loss_type == "ntxent":
                loss = loss_fn(z1, z2)
            else:  # supcon
                loss = loss_fn(z1, z2, labels)

        elif loss_type == "triplet":
            view1, view2, labels = batch
            view1 = view1.to(device)
            labels = labels.to(device)
            embeddings = model(view1)
            loss = loss_fn(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(
    model: ContrastiveModel,
    val_loader,
    loss_fn,
    loss_type: str,
    device: str,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_embeddings = []
    all_labels = []

    for batch in val_loader:
        if loss_type in ("ntxent", "supcon"):
            view1, view2, labels = batch
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)

            z1 = model(view1)
            z2 = model(view2)

            if loss_type == "ntxent":
                loss = loss_fn(z1, z2)
            else:
                loss = loss_fn(z1, z2, labels)

            # Collect encoder embeddings (not projected)
            embeddings = model.encode(view1)
            all_embeddings.append(embeddings.cpu().numpy())

        elif loss_type == "triplet":
            view1, view2, labels = batch
            view1 = view1.to(device)
            labels = labels.to(device)
            embeddings = model(view1)
            loss = loss_fn(embeddings, labels)
            all_embeddings.append(embeddings.cpu().numpy())

        all_labels.append(labels.cpu().numpy())

        total_loss += loss.item()
        num_batches += 1

    # Compute embedding quality metrics
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    from src.classifier.evaluator import compute_intra_class_cosine, compute_inter_class_cosine

    intra = compute_intra_class_cosine(embeddings, labels)
    inter = compute_inter_class_cosine(embeddings, labels)

    avg_intra = np.mean(list(intra.values())) if intra else 0.0
    avg_inter = np.mean(list(inter.values())) if inter else 1.0

    return {
        "loss": total_loss / max(num_batches, 1),
        "intra_cosine": avg_intra,
        "inter_cosine": avg_inter,
    }


def train(args):
    """Full training pipeline."""
    logger.info(f"🏋️ Training Configuration:")
    logger.info(f"  Encoder: {args.encoder}")
    logger.info(f"  Loss: {args.loss}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  LR: {args.lr}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Device: {DEVICE}")

    # Load data
    logger.info("Loading data...")
    X_train, y_train = load_split(args.train_data)
    X_val, y_val = load_split(args.val_data)

    input_dim = X_train.shape[1]
    logger.info(f"Input dim: {input_dim}, Train: {len(X_train)}, Val: {len(X_val)}")

    train_loader, val_loader = get_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        contrastive=True,
    )

    # Create model
    encoder = get_encoder(args.encoder, input_dim=input_dim, embedding_dim=args.embedding_dim)
    projection_head = ProjectionHead(input_dim=args.embedding_dim)
    model = ContrastiveModel(encoder, projection_head).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss function
    loss_fn = create_loss_fn(args.loss, args.temperature)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning rate scheduler: linear warmup + cosine annealing
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "intra_cosine": [], "inter_cosine": []}

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, args.loss, DEVICE)

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, args.loss, DEVICE)

        scheduler.step()

        elapsed = time.time() - epoch_start

        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Intra Cosine: {val_metrics['intra_cosine']:.4f} | "
            f"Inter Cosine: {val_metrics['inter_cosine']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["intra_cosine"].append(val_metrics["intra_cosine"])
        history["inter_cosine"].append(val_metrics["inter_cosine"])

        # Early stopping check
        if val_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best checkpoint
            checkpoint = {
                "epoch": epoch,
                "encoder_type": args.encoder,
                "input_dim": input_dim,
                "embedding_dim": args.embedding_dim,
                "encoder_state_dict": encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "intra_cosine": val_metrics["intra_cosine"],
                "inter_cosine": val_metrics["inter_cosine"],
                "history": history,
            }
            torch.save(checkpoint, CHECKPOINTS_DIR / "best_encoder.pt")
            logger.info(f"  💾 New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    # Save final checkpoint
    torch.save(
        {
            "epoch": epoch,
            "encoder_type": args.encoder,
            "input_dim": input_dim,
            "embedding_dim": args.embedding_dim,
            "encoder_state_dict": encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        CHECKPOINTS_DIR / "final_encoder.pt",
    )

    # Generate and save embeddings for the training set
    logger.info("Generating training embeddings for FAISS index...")
    model.eval()
    train_dataset = torch.FloatTensor(X_train).to(DEVICE)
    batch_size = 1024

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            emb = model.encode(batch)
            all_embeddings.append(emb.cpu().numpy())

    train_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(EMBEDDINGS_DIR / "train_embeddings.npy", train_embeddings)
    np.save(EMBEDDINGS_DIR / "train_labels.npy", y_train)

    # Build FAISS index
    from src.classifier.knn_classifier import FAISSKNNClassifier
    knn = FAISSKNNClassifier(embedding_dim=args.embedding_dim)
    knn.fit(train_embeddings, y_train)
    knn.save()

    logger.info("✅ Training complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {CHECKPOINTS_DIR}")
    logger.info(f"FAISS index saved to {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
