from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ContrastiveLoss",
    "InfoNCELoss",
    "SimCLRLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "SupConLoss",
    "TripletLoss",
    "get_loss_function",
]


class ContrastiveLoss(nn.Module, ABC):
    """Abstract base class for contrastive losses"""

    @abstractmethod
    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute contrastive loss between two sets of embeddings"""


class InfoNCELoss(ContrastiveLoss):
    """
    InfoNCE Loss (Noise Contrastive Estimation)
    Reference: van den Oord et al. "Representation Learning with Contrastive Predictive Coding" (2018)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = z1.size(0)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.t()) / self.temperature

        # Create labels (positive pairs are on the diagonal)
        labels = torch.arange(batch_size).to(z1.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class SimCLRLoss(ContrastiveLoss):
    """
    SimCLR Loss
    Reference: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = z1.size(0)

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # 2B x D

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.t()) / self.temperature

        # Create mask to exclude self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -float("inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_pairs = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]
        ).to(z.device)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, pos_pairs)
        return loss


class VICRegLoss(ContrastiveLoss):
    """
    VICReg Loss (Variance-Invariance-Covariance Regularization)
    Reference: Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (2022)
    """

    def __init__(
        self, lambda_param: float = 25.0, mu_param: float = 25.0, nu_param: float = 1.0
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, dim = z1.size()

        # Invariance loss (MSE between representations)
        invariance_loss = F.mse_loss(z1, z2)

        # Variance loss (encourage high variance)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        variance_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss (decorrelate features)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = torch.matmul(z1_centered.t(), z1_centered) / (batch_size - 1)
        cov_z2 = torch.matmul(z2_centered.t(), z2_centered) / (batch_size - 1)

        # Zero out diagonal
        diag_mask = torch.eye(dim, dtype=torch.bool).to(z1.device)
        cov_z1.masked_fill_(diag_mask, 0)
        cov_z2.masked_fill_(diag_mask, 0)

        covariance_loss = (cov_z1.pow(2).sum() + cov_z2.pow(2).sum()) / dim

        # Total loss
        total_loss = (
            self.lambda_param * invariance_loss
            + self.mu_param * variance_loss
            + self.nu_param * covariance_loss
        )

        return total_loss


class BarlowTwinsLoss(ContrastiveLoss):
    """
    Barlow Twins Loss
    Reference: Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (2021)
    """

    def __init__(self, lambda_param: float = 0.005):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, dim = z1.size()

        # Normalize embeddings
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-8)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-8)

        # Compute cross-correlation matrix
        c = torch.matmul(z1_norm.t(), z2_norm) / batch_size

        # Create identity matrix
        identity = torch.eye(dim).to(z1.device)

        # On-diagonal terms (should be 1)
        on_diag = torch.diagonal(c).add(-1).pow(2).sum()

        # Off-diagonal terms (should be 0)
        off_diag = (
            c.masked_fill(torch.eye(dim, dtype=torch.bool).to(z1.device), 0)
            .pow(2)
            .sum()
        )

        loss = on_diag + self.lambda_param * off_diag
        return loss


class SupConLoss(ContrastiveLoss):
    """
    Supervised Contrastive Loss
    Reference: Khosla et al. "Supervised Contrastive Learning" (2020)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if labels is None:
            # Default to InfoNCE when no labels provided
            return InfoNCELoss(self.temperature)(z1, z2)

        batch_size = z1.size(0)
        device = z1.device

        # Concatenate features and labels
        features = torch.cat([z1, z2], dim=0)
        labels = labels.repeat(2)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.t()) / self.temperature

        # Create mask for positive pairs
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()

        # Remove self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size).to(device)
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class TripletLoss(ContrastiveLoss):
    """
    Triplet Loss with hard negative mining
    Reference: Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
    """

    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = z1.size(0)

        # Compute pairwise distances
        dist_matrix = torch.cdist(z1, z2, p=2)

        # Positive distances (diagonal elements)
        pos_dist = torch.diagonal(dist_matrix)

        if self.hard_mining:
            # Hard negative mining: find hardest negatives
            mask = torch.eye(batch_size, dtype=torch.bool).to(z1.device)
            neg_dist = dist_matrix.masked_fill(mask, float("inf"))
            hard_neg_dist = torch.min(neg_dist, dim=1)[0]

            # Triplet loss with hard negatives
            loss = F.relu(pos_dist - hard_neg_dist + self.margin)
        else:
            # Use all negatives
            neg_mask = ~torch.eye(batch_size, dtype=torch.bool).to(z1.device)
            neg_dist = dist_matrix.masked_select(neg_mask).view(batch_size, -1)

            # Compute triplet loss for all combinations
            pos_dist_expanded = pos_dist.unsqueeze(1).expand(-1, batch_size - 1)
            loss = F.relu(pos_dist_expanded - neg_dist + self.margin)
            loss = loss.mean(dim=1)

        return loss.mean()


def get_loss_function(loss_type: str, **kwargs) -> ContrastiveLoss:
    """Factory function to get loss function by name"""
    loss_functions = {
        "infonce": InfoNCELoss,
        "simclr": SimCLRLoss,
        "vicreg": VICRegLoss,
        "barlow_twins": BarlowTwinsLoss,
        "supcon": SupConLoss,
        "triplet": TripletLoss,
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type](**kwargs)
