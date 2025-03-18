import neptune
from lightning.fabric import Fabric
from dataclasses import dataclass


@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int
    n_steps: int
    n_epochs: int
    batch_size: int
    n_envs: int
    truncation: int
    
    # PPO parameters
    learning_rate: float
    gamma: float
    clip_coef: float
    ent_coef: float
    max_grad_norm: float
    
    # Infrastructure settings
    checkpoint_path: str
    strategy: str
    accelerator: str
    precision: str
    devices: int
    neptune_token_path: str = "neptune_token.txt"


class NeptuneLogger:
    """Handles logging experiment metrics to Neptune."""

    def __init__(self, config: SelfPlayConfig, fabric: Fabric):
        self.fabric = fabric
        self.config = config
        
        # Only initialize Neptune on the main process
        if self.fabric.is_global_zero:
            # Load Neptune API key from file
            with open(self.config.neptune_token_path, "r") as f:
                neptune_key = f.read().strip()
                
            self.run = neptune.init_run(
                project="strakam/selfplay",
                name="generals-selfplay",
                tags=["selfplay", "training"],
                api_token=neptune_key,
            )
        
        self._log_config()

    def _log_config(self):
        """Log experiment configuration parameters."""
        if self.fabric.is_global_zero:
            self.run["parameters"] = {
                # Training hyperparameters
                "training_iterations": self.config.training_iterations,
                "n_steps": self.config.n_steps,
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
                "n_envs": self.config.n_envs,
                # Environment settings
                "truncation": self.config.truncation,
                # PPO hyperparameters
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "clip_coef": self.config.clip_coef,
                "ent_coef": self.config.ent_coef,
                "max_grad_norm": self.config.max_grad_norm,
                # Infrastructure settings
                "checkpoint_path": self.config.checkpoint_path,
                "strategy": self.config.strategy,
                "accelerator": self.config.accelerator,
                "precision": self.config.precision,
                "devices": self.config.devices,
            }

    def log_metrics(self, metrics: dict):
        """Logs a batch of metrics to Neptune."""
        if self.fabric.is_global_zero:
            for key, value in metrics.items():
                self.run[f"{key}"].log(value)

    def close(self):
        """Close the Neptune run."""
        if self.fabric.is_global_zero:
            self.run.stop() 