"""
Contrastive Losses — NT-Xent, Supervised Contrastive, and Triplet Loss.

NT-Xent (Normalized Temperature-scaled Cross-Entropy) is the primary loss
from SimCLR. It maximizes agreement between two augmented views of the
same flow while pushing apart views of different flows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TEMPERATURE, TRIPLET_MARGIN


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (SimCLR).

    For a batch of N flows, we create 2N views (2 augmentations per flow).
    The positive pair for view i is view i+N (same flow, different augmentation).
    All other 2(N-1) views in the batch are negatives.

    Loss = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
    where (i, j) is a positive pair and k iterates over all negatives.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Projected embeddings from view 1 [batch_size, dim]
            z2: Projected embeddings from view 2 [batch_size, dim]

        Returns:
            Scalar NT-Xent loss.
        """
        batch_size = z1.size(0)
        device = z1.device

        # Concatenate both views: [2N, dim]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        # Compute all-pairs cosine similarity: [2N, 2N]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix.masked_fill_(mask, -1e9)

        # Positive pair labels:
        # For z1[i], positive is z2[i] at index i+N
        # For z2[i], positive is z1[i] at index i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device),
        ])

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Extends NT-Xent to use class labels — all flows of the same class
    are treated as positives for each other, not just augmented views.

    This leverages label information during contrastive pretraining,
    which can produce better-separated embeddings for known classes.

    Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1: Projected embeddings from view 1 [batch_size, dim]
            z2: Projected embeddings from view 2 [batch_size, dim]
            labels: Class labels [batch_size]

        Returns:
            Scalar SupCon loss.
        """
        batch_size = z1.size(0)
        device = z1.device

        # Concatenate views: [2N, dim]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        # Duplicate labels for both views
        labels = torch.cat([labels, labels], dim=0)  # [2N]

        # Similarity matrix: [2N, 2N]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask for positive pairs: same class, different sample
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2N, 2N]
        self_mask = ~torch.eye(2 * batch_size, device=device).bool()
        positive_mask = label_matrix & self_mask

        # For numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Log-sum-exp over negatives
        exp_sim = torch.exp(sim_matrix) * self_mask.float()
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positives
        log_prob = sim_matrix - log_sum_exp
        positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1)

        # Normalize by number of positives per anchor
        num_positives = positive_mask.float().sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)

        loss = -(positive_log_prob / num_positives).mean()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining.

    For each anchor, find the hardest positive (same class, farthest)
    and hardest negative (different class, closest), then optimize:
    loss = max(0, d(a, p) - d(a, n) + margin)

    Simpler to debug than NT-Xent. Use as fallback.
    """

    def __init__(self, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, dim]
            labels: Class labels [batch_size]

        Returns:
            Scalar triplet loss with hard mining.
        """
        embeddings = F.normalize(embeddings, dim=1)

        # Pairwise distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Masks for positive and negative pairs
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = ~torch.eye(len(labels), device=labels.device).bool()

        positive_mask = labels_eq & self_mask
        negative_mask = ~labels_eq

        # Hard positive: max distance among positives
        positive_dist = dist_matrix * positive_mask.float()
        hardest_positive, _ = positive_dist.max(dim=1)

        # Hard negative: min distance among negatives
        negative_dist = dist_matrix.clone()
        negative_dist[~negative_mask] = float("inf")
        hardest_negative, _ = negative_dist.min(dim=1)

        # Triplet loss
        loss = F.relu(hardest_positive - hardest_negative + self.margin)
        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center Loss — Pulls same-class embeddings toward learned class centroids.

    This directly optimizes intra-class compactness (the failing KPI).
    Combined with SupCon (which optimizes inter-class separation),
    we get embeddings that are both compact within classes AND separated between classes.

    Reference: Wen et al., "A Discriminative Feature Learning Approach
    for Deep Face Recognition" (ECCV 2016)
    """

    def __init__(self, num_classes: int, embedding_dim: int, alpha: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Center update rate
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, dim]
            labels: Class labels [batch_size]

        Returns:
            Center loss scalar.
        """
        # Get centers for each sample's class
        batch_centers = self.centers[labels]  # [batch_size, dim]

        # Compute squared distance to class center
        loss = ((embeddings - batch_centers) ** 2).sum(dim=1).mean()

        return loss


class CentroidRepulsionLoss(nn.Module):
    """
    Centroid Repulsion Loss — Pushes class centroids apart.

    Directly optimizes inter-class cosine similarity by computing
    batch-level class centroids and penalizing high pairwise cosine
    similarity between them.

    Unlike SupCon (which operates on sample pairs), this operates
    on class-level structure, making it much more efficient at
    reducing inter-class cosine similarity.
    """

    def __init__(self, margin: float = 0.0):
        """
        Args:
            margin: Target maximum cosine similarity between centroids.
                    Loss is applied when cos_sim > margin.
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings [batch_size, dim]
            labels: Class labels [batch_size]

        Returns:
            Repulsion loss scalar.
        """
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute class centroids from batch
        centroids = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(dim=0)
                centroid = F.normalize(centroid, dim=0)  # Re-normalize
                centroids.append(centroid)

        centroids = torch.stack(centroids)  # [num_classes_in_batch, dim]

        # Pairwise cosine similarity between centroids
        sim_matrix = centroids @ centroids.T  # [K, K]

        # Only penalize upper triangle (avoid double counting + diagonal)
        K = len(centroids)
        mask = torch.triu(torch.ones(K, K, device=embeddings.device), diagonal=1).bool()
        pairwise_sims = sim_matrix[mask]

        # Hinge loss: penalize similarity above margin
        loss = F.relu(pairwise_sims - self.margin).mean()

        return loss


class CombinedContrastiveLoss(nn.Module):
    """
    Combined loss: SupCon + CenterLoss + optional CrossEntropy.

    - SupCon: maximizes inter-class separation (pushes clusters apart)
    - CenterLoss: maximizes intra-class compactness (pulls clusters tight)
    - CrossEntropy: adds direct classification signal (improves accuracy)

    This is the key innovation for achieving BOTH high intra-class cosine
    AND high accuracy simultaneously.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        temperature: float = 0.5,
        center_weight: float = 0.1,
        ce_weight: float = 0.5,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.center_loss = CenterLoss(num_classes=num_classes, embedding_dim=embedding_dim)
        self.ce_head = nn.Linear(embedding_dim, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

        self.center_weight = center_weight
        self.ce_weight = ce_weight

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        h1: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1, z2: Projected embeddings [batch, proj_dim] (for SupCon)
            h1: Encoder embeddings [batch, embed_dim] (for center + CE)
            labels: Class labels [batch]
        """
        # SupCon loss (separation)
        loss_supcon = self.supcon(z1, z2, labels)

        # Center loss (compactness)
        h1_norm = F.normalize(h1, dim=1)
        loss_center = self.center_loss(h1_norm, labels)

        # Cross-entropy loss (accuracy)
        logits = self.ce_head(h1)
        loss_ce = self.ce_loss(logits, labels)

        total = loss_supcon + self.center_weight * loss_center + self.ce_weight * loss_ce
        return total
