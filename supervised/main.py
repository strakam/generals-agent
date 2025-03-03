from dataclasses import dataclass, field
from pathlib import Path
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger

from dataloader import ReplayDataset, per_worker_init_fn, collate_fn
from network import Network


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Data parameters
    dataset_name: str = "above70"
    n_samples: int = 11_370_000
    buffer_size: int = 18000
    batch_size: int = 1920
    n_workers: int = 32
    # Training parameters
    learning_rate: float = 2e-4
    n_epochs: int = 16
    clip_val: float = 2.0
    seed: int = 42
    channel_sequence: list[int] = field(default_factory=lambda: [256, 256, 288, 288])
    repeats: list[int] = field(default_factory=lambda: [2, 2, 2, 1])

    # Logging and checkpointing
    log_every_n_steps: int = 20
    checkpoint_every_n_steps: int = 4000
    checkpoint_dir: str = "/storage/praha1/home/strakam3/sup_checkpoints/"
    neptune_token_path: str = "neptune_token.txt"
    model_ckpt: str = None

    def __post_init__(self):
        """Calculate dependent parameters after initialization."""
        self.max_steps = self.n_samples // self.batch_size * self.n_epochs


class DataModule:
    """Handles dataset and dataloader setup."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup_dataset(self) -> torch.utils.data.DataLoader:
        """Create and configure the dataset and dataloader."""
        dataset_path = Path("datasets") / self.config.dataset_name
        replays = [str(dataset_path / name) for name in os.listdir(dataset_path)]

        # Shuffle replays
        torch.randperm(len(replays))

        # Create dataset
        dataset = ReplayDataset(replays, buffer_size=self.config.buffer_size)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            worker_init_fn=per_worker_init_fn,
            collate_fn=collate_fn,
        )


class TrainingModule:
    """Handles training setup and execution."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_environment()

    def _setup_environment(self):
        """Configure PyTorch and training environment."""
        torch.manual_seed(self.config.seed)
        torch.set_float32_matmul_precision("high")

    def _create_logger(self) -> NeptuneLogger:
        """Initialize Neptune logger."""
        with open(self.config.neptune_token_path, "r") as f:
            neptune_key = f.read().strip()

        return NeptuneLogger(
            api_key=neptune_key,
            project="strakam/supervised-agent",
        )

    def _create_checkpoint_callback(self) -> ModelCheckpoint:
        """Configure checkpoint callback."""
        return ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename="{step}",
            save_top_k=-1,
            every_n_train_steps=self.config.checkpoint_every_n_steps,
        )

    def create_trainer(self) -> L.Trainer:
        """Create and configure the Lightning trainer."""
        return L.Trainer(
            logger=self._create_logger(),
            log_every_n_steps=self.config.log_every_n_steps,
            max_steps=self.config.max_steps,
            max_epochs=-1,
            gradient_clip_val=self.config.clip_val,
            gradient_clip_algorithm="norm",
            callbacks=[self._create_checkpoint_callback()],
        )


def main():
    # Initialize configuration
    config = TrainingConfig()

    # Setup data
    data_module = DataModule(config)
    dataloader = data_module.setup_dataset()

    # Setup training
    training_module = TrainingModule(config)
    trainer = training_module.create_trainer()

    # Start training
    if config.model_ckpt:
        # Load checkpoint but override learning rate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Network(lr=config.learning_rate, compile=False)
        state_dict = torch.load(config.model_ckpt, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)
        trainer.fit(model, train_dataloaders=dataloader)
    else:
        model = Network(
            lr=config.learning_rate, channel_sequence=config.channel_sequence, repeats=config.repeats, compile=True
        )
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        trainer.logger.experiment["total_parameters"] = sum(p.numel() for p in model.parameters())
        trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
