"""
PyTorch Dataset classes for flow-based traffic classification.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from src.data.augmentations import FlowAugmentor, generate_contrastive_pair


class FlowDataset(Dataset):
    """
    Standard dataset for supervised classification.
    Returns (features, label) pairs.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Args:
            features: Flow feature matrix [N, feature_dim].
            labels: Integer class labels [N].
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ContrastiveFlowDataset(Dataset):
    """
    Dataset for contrastive learning (NT-Xent / SimCLR).
    Returns (view1, view2, label) — two augmented views of the same flow.

    During training, the NT-Xent loss uses the batch to form positive pairs
    (view1_i, view2_i) and negative pairs (view_i, view_j where i ≠ j).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augmentor: Optional[FlowAugmentor] = None,
    ):
        """
        Args:
            features: Flow feature matrix [N, feature_dim].
            labels: Integer class labels [N].
            augmentor: FlowAugmentor instance for generating views.
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augmentor = augmentor or FlowAugmentor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        view1, view2 = generate_contrastive_pair(x, self.augmentor)
        return view1, view2, self.labels[idx]


class SupervisedContrastiveDataset(Dataset):
    """
    Dataset for Supervised Contrastive Learning (SupCon).
    Same as ContrastiveFlowDataset but the loss function uses labels
    to identify all positive pairs within a batch (all samples of the same class).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augmentor: Optional[FlowAugmentor] = None,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augmentor = augmentor or FlowAugmentor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        view1, view2 = generate_contrastive_pair(x, self.augmentor)
        return view1, view2, self.labels[idx]


def load_split(split_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-saved train/val/test split CSV.

    Returns:
        Tuple of (features, labels) as numpy arrays.
    """
    df = pd.read_csv(split_path)
    labels = df["label"].values
    features = df.drop(columns=["label"]).values
    return features.astype(np.float32), labels.astype(np.int64)


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 512,
    contrastive: bool = True,
    num_workers: int = 0,
) -> Tuple:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        batch_size: Batch size (larger = more negatives for contrastive).
        contrastive: If True, use ContrastiveFlowDataset.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from torch.utils.data import DataLoader

    if contrastive:
        train_dataset = ContrastiveFlowDataset(X_train, y_train)
        val_dataset = ContrastiveFlowDataset(X_val, y_val, augmentor=FlowAugmentor(augmentation_prob=0.0))
    else:
        train_dataset = FlowDataset(X_train, y_train)
        val_dataset = FlowDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,   # Important for contrastive learning — need consistent batch sizes
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
