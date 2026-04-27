"""
Projection Head — MLP projection for contrastive learning.

The projection head maps encoder outputs to a space where the contrastive
loss is applied. This is used during training only — at inference time,
we use the encoder output directly as the embedding.

This follows the SimCLR finding that a non-linear projection head
significantly improves the quality of learned representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import EMBEDDING_DIM


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head (SimCLR style).

    Maps encoder output → projection space → L2-normalized embeddings.
    The contrastive loss operates in this projected space.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder output [batch, input_dim]

        Returns:
            L2-normalized projection [batch, output_dim]
        """
        projected = self.mlp(x)
        return F.normalize(projected, dim=1)


class ContrastiveModel(nn.Module):
    """
    Full contrastive learning model: Encoder + Projection Head.

    During training: use model.forward() to get projected embeddings for NT-Xent.
    During inference: use model.encode() to get encoder embeddings for downstream tasks.
    """

    def __init__(self, encoder: nn.Module, projection_head: ProjectionHead):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through encoder + projection head."""
        h = self.encoder(x)            # Encoder embedding
        z = self.projection_head(h)    # Projected embedding (for loss)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode-only pass (no projection head).
        Use this for downstream tasks (k-NN, SVM, visualization).
        """
        with torch.no_grad():
            h = self.encoder(x)
            return F.normalize(h, dim=1)  # L2-normalize for cosine similarity
