import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model architecture"""

    fl_encoder_model: str = "microsoft/codebert-base"
    nl_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    projection_dim: int = 256
    pooling_strategy: str = "cls"
    freeze_base_models: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""

    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    save_every: int = 10
    eval_every: int = 5
    early_stopping_patience: int = 10


@dataclass
class LossConfig:
    """Configuration for loss function"""

    loss_type: str = "infonce"
    temperature: float = 0.07

    # VICReg parameters
    vicreg_lambda: float = 25.0
    vicreg_mu: float = 25.0
    vicreg_nu: float = 1.0

    # Barlow Twins parameters
    barlow_lambda: float = 0.005

    # Triplet loss parameters
    triplet_margin: float = 0.3
    triplet_hard_mining: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""

    dataset_name: str = "frenzy"
    data_path: str = "data/raw/frenzy_math.json"
    max_length: int = 512
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment metadata
    experiment_name: str = "joint_embedding_v1"
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "cuda"

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
