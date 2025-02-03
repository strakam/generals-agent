import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
from generals import GridFactory, GymnasiumGenerals
import gymnasium as gym
import argparse
from modules.network import load_network
from modules.agent import SelfPlayAgent
import neptune
from lightning.fabric import Fabric
import pytorch_lightning as L
from torch.utils.data import DistributedSampler, RandomSampler, BatchSampler


@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int = 1000
    n_envs: int = 2
    n_steps: int = 10
    batch_size: int = 8
    n_epochs: int = 2
    truncation: int = 1500
    checkpoint_path: str = "step=52000.ckpt"

    # PPO parameters
    gamma: float = 1.0
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    clip_coef: float = 0.2
    ent_coef: float = 0.01

    # Lightning fabric parameters
    strategy: str = "auto"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42


class NeptuneLogger:
    """Handles logging experiment metrics to Neptune."""

    def __init__(self, config: SelfPlayConfig, fabric: Fabric):
        # Add fabric parameter
        self.fabric = fabric
        # Only initialize Neptune on the main process
        if self.fabric.is_global_zero:
            self.run = neptune.init_run(
                project="strakam/selfplay",
                name="generals-selfplay",
                tags=["selfplay", "training"],
            )
        self.config = config
        self._log_config()

    def _log_config(self):
        """Log experiment configuration parameters."""
        if self.fabric.is_global_zero:
            self.run["parameters"] = {
                "n_envs": self.config.n_envs,
                "training_iterations": self.config.training_iterations,
                "n_steps": self.config.n_steps,
                "truncation": self.config.truncation,
                "checkpoint_path": self.config.checkpoint_path,
                "strategy": self.config.strategy,
                "accelerator": self.config.accelerator,
                "devices": self.config.devices,
            }

    def close(self):
        """Close the Neptune run."""
        if self.fabric.is_global_zero:
            self.run.stop()


def create_environment(agent_names: List[str], config: SelfPlayConfig) -> gym.vector.AsyncVectorEnv:
    # Generate grid factory as in real generals.io
    grid_factory = GridFactory(mode="generalsio")
    return gym.vector.AsyncVectorEnv(
        [
            lambda: GymnasiumGenerals(
                agents=agent_names,
                grid_factory=grid_factory,
                truncation=config.truncation,
                pad_observations_to=24,
            )
            for _ in range(config.n_envs)
        ],
    )


def train(
    fabric: Fabric,
    network: L.LightningModule,
    optimizer: torch.optim.Optimizer,
    data: dict[str, torch.Tensor],
    global_step: int,
    cfg: SelfPlayConfig,
):
    # Get actual data size
    total_size = data["observations"].shape[0]
    indices = torch.arange(total_size)

    effective_batch_size = min(cfg.batch_size, total_size)

    # Choose sampler based on distributed setting
    if fabric.world_size > 1:
        sampler = DistributedSampler(
            indices,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=cfg.seed,
        )
    else:
        sampler = RandomSampler(indices)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=effective_batch_size,
        drop_last=False,
    )

    network.train()
    for epoch in range(cfg.n_epochs):
        # Set epoch for distributed sampler
        if fabric.world_size > 1:
            sampler.set_epoch(epoch)

        for batch_indices in batch_sampler:
            # Convert indices list to tensor for indexing
            batch_indices_tensor = torch.tensor(batch_indices, device=fabric.device)
            # Index the data using tensor indexing
            batch = {k: v[batch_indices_tensor] for k, v in data.items()}
            loss = network.ppo_loss(batch, cfg)

            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(network, optimizer, max_norm=cfg.max_grad_norm)
            optimizer.step()
            # print loss
            fabric.print(f"Loss: {loss.item()}")

        # network.on_train_epoch_end(global_step)

def main(args):
    # Initialize hyperparameters
    cfg = SelfPlayConfig()

    # Initialize Fabric - removed DDPStrategy since we're using "auto"
    fabric = Fabric(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
    )
    fabric.launch()

    fabric.seed_everything(cfg.seed)

    # Initialize logger after fabric launch
    logger = NeptuneLogger(cfg, fabric)

    # Load agent
    n_obs = 2 * cfg.n_envs
    network = load_network(cfg.checkpoint_path)
    optimizer, _ = network.configure_optimizers(lr=cfg.learning_rate)
    # Setup the network with Fabric
    network, optimizer = fabric.setup(network, optimizer)
    agent = SelfPlayAgent(network=network, batch_size=n_obs, device=fabric.device)

    # Setup environment
    agent_names = ["1", "2"]
    envs = create_environment(agent_names, cfg)

    n_channels = agent.n_channels

    obs_shape = (cfg.n_steps, 2 * cfg.n_envs, n_channels, 24, 24)
    masks_shape = (cfg.n_steps, 2 * cfg.n_envs, 24, 24, 4)
    actions_shape = (cfg.n_steps, 2 * cfg.n_envs, 5)
    logprobs_shape = (cfg.n_steps, 2 * cfg.n_envs)
    rewards_shape = (cfg.n_steps, 2 * cfg.n_envs)
    dones_shape = (cfg.n_steps, cfg.n_envs)

    # Initialize tensors using fabric's device context
    with fabric.device:
        obs = torch.zeros(obs_shape, dtype=torch.float16)
        actions = torch.zeros(actions_shape, dtype=torch.float16)
        logprobs = torch.zeros(logprobs_shape)
        rewards = torch.zeros(rewards_shape, dtype=torch.float16)
        dones = torch.zeros(dones_shape, dtype=torch.bool)
        masks = torch.zeros(masks_shape, dtype=torch.bool)

    global_step = 0

    def process_observations(obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        with fabric.device:
            obs_tensor = torch.from_numpy(obs).half()
            # remove the agent dimension and stack it into the first dimension
            reshaped_obs = obs_tensor.reshape(cfg.n_envs * 2, -1, 24, 24)
            augmented_obs = agent.augment_observation(reshaped_obs)

            # Fix mask stacking to ensure correct shape
            mask = (
                torch.from_numpy(np.stack([info[4] for agent in agent_names for info in infos[agent]]))
                .bool()
                .reshape(2 * cfg.n_envs, 24, 24, 4)
            )  # Explicitly reshape to ensure correct dimensions
            _rewards = torch.tensor([info[5] for agent in agent_names for info in infos[agent]])
        return augmented_obs, mask, _rewards

    next_obs, infos = envs.reset()
    next_obs, mask, _rewards = process_observations(next_obs, infos)
    with fabric.device:
        next_done = torch.zeros(cfg.n_envs, dtype=torch.bool)


    for iteration in range(1, cfg.training_iterations + 1):
        for step in range(0, cfg.n_steps):
            global_step += cfg.n_envs
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = mask

            with torch.no_grad():
                actions[step], logprobs[step], _ = network.get_action(next_obs, mask)
            _actions = actions[step].reshape(cfg.n_envs, 2, -1).cpu().numpy().astype(int)
            next_obs, _, terminations, truncations, infos = envs.step(_actions)
            next_obs, mask, _rewards = process_observations(next_obs, infos)
            rewards[step] = _rewards
            next_done = np.logical_or(terminations, truncations)
            with fabric.device:
                next_done = torch.tensor(next_done, dtype=torch.bool)

        with torch.no_grad(), fabric.device:
            returns = torch.zeros_like(rewards)
            next_value = torch.zeros_like(rewards[0])
            next_non_terminal = torch.repeat_interleave(1.0 - next_done.float(), 2, dim=0)

            # Calculate returns and advantages
            for t in reversed(range(cfg.n_steps)):
                # Calculate returns with discounting for terminal states
                returns[t] = rewards[t] + cfg.gamma * next_value * next_non_terminal
                next_value = returns[t]
                # Broadcast dones to match the shape of rewards (2 agents per env)
                next_non_terminal = torch.repeat_interleave(1.0 - dones[t].float(), 2, dim=0)

        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_masks = masks.reshape(-1, *masks.shape[2:])

        # Store flattened tensors in dictionary for training
        training_data = {
            "observations": b_obs,
            "actions": b_actions,
            "logprobs": b_logprobs,
            "returns": b_returns,
            "masks": b_masks,
        }

        train(fabric, network, optimizer, training_data, global_step, cfg)

        # Use fabric.print instead of print for proper distributed logging
        fabric.print(f"Iteration {iteration} completed")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
