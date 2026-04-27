"""
Augmentations — Flow-level data augmentations for contrastive learning.

These create two "views" of the same flow to form positive pairs for NT-Xent.
"""

import torch
import numpy as np
from typing import Tuple

from src.config import AUG_NOISE_STD, AUG_MASK_RATIO, AUG_JITTER_RATIO


class FlowAugmentor:
    """
    Applies a composition of augmentations to flow feature vectors
    to generate positive pairs for contrastive learning.

    Each augmentation slightly perturbs the flow features while preserving
    the overall traffic pattern semantics. Two different augmentations of
    the same flow should still be recognizable as the same "type" of traffic.
    """

    def __init__(
        self,
        noise_std: float = AUG_NOISE_STD,
        mask_ratio: float = AUG_MASK_RATIO,
        jitter_ratio: float = AUG_JITTER_RATIO,
        augmentation_prob: float = 0.8,
    ):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.jitter_ratio = jitter_ratio
        self.augmentation_prob = augmentation_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random subset of augmentations to a flow feature vector."""
        augmented = x.clone()

        if np.random.random() < self.augmentation_prob:
            augmented = self.gaussian_noise(augmented)

        if np.random.random() < self.augmentation_prob:
            augmented = self.feature_masking(augmented)

        if np.random.random() < self.augmentation_prob:
            augmented = self.temporal_jitter(augmented)

        if np.random.random() < self.augmentation_prob * 0.5:
            augmented = self.feature_scaling(augmented)

        return augmented

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to all features."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def feature_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask (zero out) a fraction of features."""
        mask = torch.ones_like(x)
        num_mask = int(x.shape[-1] * self.mask_ratio)
        indices = torch.randperm(x.shape[-1])[:num_mask]
        mask[..., indices] = 0.0
        return x * mask

    def temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Perturb values by a small multiplicative factor (simulates timing variation)."""
        jitter = 1.0 + (torch.randn_like(x) * self.jitter_ratio)
        return x * jitter

    def feature_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random uniform scaling of all features (simulates bandwidth variation)."""
        scale = 0.8 + 0.4 * torch.rand(1)  # Scale between 0.8 and 1.2
        return x * scale

    def feature_shuffle(self, x: torch.Tensor, group_size: int = 4) -> torch.Tensor:
        """
        Shuffle features within groups (preserving intra-group relationships).
        Simulates slight reordering of related metrics.
        """
        augmented = x.clone()
        num_features = x.shape[-1]

        for start in range(0, num_features - group_size, group_size):
            end = min(start + group_size, num_features)
            perm = torch.randperm(end - start) + start
            augmented[..., start:end] = x[..., perm]

        return augmented


def generate_contrastive_pair(
    x: torch.Tensor,
    augmentor: FlowAugmentor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two augmented views of the same flow for contrastive learning.

    Args:
        x: Original flow feature vector [feature_dim] or [batch, feature_dim].
        augmentor: FlowAugmentor instance.

    Returns:
        Tuple of (view1, view2), each augmented differently.
    """
    view1 = augmentor(x)
    view2 = augmentor(x)
    return view1, view2
