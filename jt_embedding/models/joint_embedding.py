import torch
import torch.nn as nn

from ai_for_math.models.encoders import TransformerEncoder
from ai_for_math.models.losses import (
    BarlowTwinsLoss,
    ContrastiveLoss,
    InfoNCELoss,
    SimCLRLoss,
    VICRegLoss,
)
from ai_for_math.training.config import ExperimentConfig, LossConfig


class JointEmbeddingModel(nn.Module):
    """Joint embedding model for formal and informal mathematical theorems"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        # Initialize encoders
        self.fl_encoder = TransformerEncoder(
            config.model.fl_encoder_model, config.model.projection_dim
        )
        self.nl_encoder = TransformerEncoder(
            config.model.nl_encoder_model, config.model.projection_dim
        )

        # Initialize loss function
        self.loss_fn = self._get_loss_function(config.loss.loss_type, config.loss)

    def _get_loss_function(self, loss_type: str, config: LossConfig) -> ContrastiveLoss:
        """Get the appropriate loss function"""
        if loss_type == "infonce":
            return InfoNCELoss(temperature=config.temperature)
        elif loss_type == "simclr":
            return SimCLRLoss(temperature=config.temperature)
        elif loss_type == "vicreg":
            return VICRegLoss(config.vicreg_lambda, config.vicreg_mu, config.vicreg_nu)
        elif loss_type == "barlow_twins":
            return BarlowTwinsLoss(config.barlow_lambda)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Encode formal theorems
        fl_embeddings = self.fl_encoder(
            batch["formal_input_ids"], batch["formal_attention_mask"]
        )

        # Encode informal theorems
        nl_embeddings = self.nl_encoder(
            batch["informal_input_ids"], batch["informal_attention_mask"]
        )

        # Compute loss
        loss = self.loss_fn(fl_embeddings, nl_embeddings)

        return {
            "loss": loss,
            "fl_embeddings": fl_embeddings,
            "nl_embeddings": nl_embeddings,
        }

    def encode_formal(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode formal theorem"""
        return self.fl_encoder(input_ids, attention_mask)

    def encode_informal(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode informal theorem"""
        return self.nl_encoder(input_ids, attention_mask)
