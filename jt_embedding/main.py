import logging

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ai_for_math.data import dataset as math_dataset
from ai_for_math.models import joint_embedding
from ai_for_math.training import config as exp_config
from ai_for_math.training import trainer as exp_trainer


def main() -> None:
    """Main function to run the joint embedding model training"""
    # Load experiment configuration
    config = exp_config.ExperimentConfig()
    fl_tokenizer = AutoTokenizer.from_pretrained(config.model.fl_encoder_model)
    nl_tokenizer = AutoTokenizer.from_pretrained(config.model.nl_encoder_model)

    dataset = math_dataset.FrenzyMathDataset.from_datasets(
        fl_tokenizer=fl_tokenizer,
        nl_tokenizer=nl_tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
    )
    # Initialize model
    joint_embedding_model = joint_embedding.JointEmbeddingModel(config=config)

    # Initialize trainer
    trainer = exp_trainer.JointEmbeddingTrainer(
        model=joint_embedding_model, config=config
    )
    trainer.train_epoch(dataloader=dataloader)
    # Here you would typically load your data and start training
    # For example:
    for epoch in range(config.training.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")
        trainer.train_epoch(dataloader=dataloader)
        trainer.evaluate(dataloader=dataloader)


if __name__ == "__main__":
    main()
