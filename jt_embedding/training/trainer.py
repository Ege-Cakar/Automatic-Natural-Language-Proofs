import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_for_math.models.joint_embedding import JointEmbeddingModel
from ai_for_math.training.config import ExperimentConfig


class JointEmbeddingTrainer:
    """Trainer for joint embedding model"""

    def __init__(self, model: JointEmbeddingModel, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.training.learning_rate
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
        for batch_idx, batch in prog_bar:
            assert isinstance(batch, dict), "Batch must be a dictionary"
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Forward pass
            outputs = self.model(batch)
            loss = outputs["loss"]
            assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(
                    "Batch %d/%d, Loss: %.4f", batch_idx, len(dataloader), loss.item()
                )

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                total_loss += outputs["loss"].item()

        return total_loss / len(dataloader)

    def compute_similarity(self, formal_text: str, informal_text: str) -> float:
        """Compute similarity between formal and informal theorem"""
        self.model.eval()

        # Tokenize
        fl_tokens = self.model.fl_encoder.model.tokenizer(
            formal_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        nl_tokens = self.model.nl_encoder.model.tokenizer(
            informal_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            fl_emb = self.model.encode_formal(
                fl_tokens["input_ids"], fl_tokens["attention_mask"]
            )
            nl_emb = self.model.encode_informal(
                nl_tokens["input_ids"], nl_tokens["attention_mask"]
            )

            similarity = F.cosine_similarity(x1=fl_emb, x2=nl_emb, dim=1)

        return similarity.item()
