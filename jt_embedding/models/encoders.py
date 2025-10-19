from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, T5EncoderModel


__all__ = [
    "PoolingStrategy",
    "AbstractEncoder",
    "TransformerEncoder",
    "CodeBERTEncoder",
    "MathBERTEncoder",
    "SentenceTransformerEncoder",
    "MockEncoder",
]


class PoolingStrategy(Enum):
    """Pooling strategies for transformer outputs.

    CLS: Use the [CLS] token representation.
    MEAN: Average pooling over all token representations.
    MAX: Max pooling over all token representations.
    """

    CLS = "cls"
    MEAN = "mean"
    MAX = "max"


class AbstractEncoder(nn.Module, ABC):
    """Abstract base class for encoders"""

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning embeddings"""

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension"""


class TransformerEncoder(AbstractEncoder):
    """Transformer-based encoder using HuggingFace models"""

    def __init__(
        self,
        model_name: str,
        projection_dim: int = 256,
        pooling_strategy: str = "cls",
        freeze_base: bool = False,
    ):
        super().__init__()
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size
        self.projection_dim = projection_dim
        self.pooling_strategy = PoolingStrategy(pooling_strategy)

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, projection_dim),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply pooling strategy
        if self.pooling_strategy == PoolingStrategy.CLS:
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling_strategy == PoolingStrategy.MEAN:
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling_strategy == PoolingStrategy.MAX:
            embeddings = self._max_pooling(outputs.last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Apply projection
        projected = self.projection(embeddings)
        return F.normalize(projected, p=2, dim=1)

    def _mean_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling with attention mask"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        return torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _max_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Max pooling with attention mask"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        hidden_states[input_mask_expanded == 0] = -1e9
        return torch.max(hidden_states, 1)[0]

    def get_embedding_dim(self) -> int:
        return self.projection_dim


class CodeBERTEncoder(TransformerEncoder):
    """Specialized encoder for code/formal language"""

    def __init__(self, projection_dim: int = 256):
        super().__init__(
            model_name="microsoft/codebert-base",
            projection_dim=projection_dim,
            pooling_strategy="cls",
        )


class MathBERTEncoder(TransformerEncoder):
    """Specialized encoder for mathematical language"""

    def __init__(self, projection_dim: int = 256):
        super().__init__(
            model_name="tbs17/MathBERT",
            projection_dim=projection_dim,
            pooling_strategy="cls",
        )


class SentenceTransformerEncoder(TransformerEncoder):
    """Encoder using sentence transformers"""

    def __init__(self, projection_dim: int = 256, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(
            model_name=f"sentence-transformers/{model_name}",
            projection_dim=projection_dim,
            pooling_strategy="mean",
        )


class MockEncoder(AbstractEncoder):
    """Mock encoder for testing purposes.

    Also it should have parameters attribute to be compatible with the training loop.
    Returns random embeddings of fixed dimension.
    """

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.randn(embedding_dim), requires_grad=True)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.embeddings.unsqueeze(0).expand(input_ids.size(0), -1)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
