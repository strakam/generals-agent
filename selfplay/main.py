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
    n_envs: int = 4
    n_steps: int = 320
    batch_size: int = 16
    n_epochs: int = 4
    truncation: int = 1500
    grid_size: int = 8
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
    min_grid_dims = (config.grid_size, config.grid_size)
    max_grid_dims = (config.grid_size, config.grid_size)
    # TODO: here we should use the generalsio grid factory when legit training
    # grid_factory = GridFactory(mode="generalsio", min_grid_dims=min_grid_dims, max_grid_dims=max_grid_dims)
    grid_factory = GridFactory(min_grid_dims=min_grid_dims, max_grid_dims=max_grid_dims)
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
    n_obs = cfg.n_envs
    network = load_network(cfg.checkpoint_path)
    optimizer, _ = network.configure_optimizers(lr=cfg.learning_rate)
    # Setup the network with Fabric
    network, optimizer = fabric.setup(network, optimizer)
    agent = SelfPlayAgent(network=network, batch_size=n_obs, device=fabric.device)

    # Setup environment
    agent_names = ["1", "2"]
    envs = create_environment(agent_names, cfg)

    n_channels = agent.n_channels

    # Update tensor shapes to have explicit agent dimension
    n_agents = 2
    obs_shape = (cfg.n_steps, cfg.n_envs, n_agents, n_channels, 24, 24)
    masks_shape = (cfg.n_steps, cfg.n_envs, n_agents, 24, 24, 4)
    actions_shape = (cfg.n_steps, cfg.n_envs, n_agents, 5)
    logprobs_shape = (cfg.n_steps, cfg.n_envs, n_agents)
    rewards_shape = (cfg.n_steps, cfg.n_envs, n_agents)
    dones_shape = (cfg.n_steps, cfg.n_envs)  # Keeps same shape as it's per-environment

    # Initialize tensors using fabric's device context
    with fabric.device:
        obs = torch.zeros(obs_shape, dtype=torch.float16)
        actions = torch.zeros(actions_shape, dtype=torch.float16)
        logprobs = torch.zeros(logprobs_shape)
        rewards = torch.zeros(rewards_shape, dtype=torch.float16)
        dones = torch.zeros(dones_shape, dtype=torch.bool)
        masks = torch.zeros(masks_shape, dtype=torch.bool)

    global_step = 0

    def process_observations(obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with fabric.device:
            # Assume obs comes in with shape (n_envs, 2, channels, 24, 24)
            obs_tensor = torch.from_numpy(obs).half()

            # Process each agent's observations separately
            augmented_obs = []
            for agent_idx in range(n_agents):
                agent_obs = obs_tensor[:, agent_idx, ...]
                augmented_agent_obs = agent.augment_observation(agent_obs)
                augmented_obs.append(augmented_agent_obs)

            # Stack along agent dimension (n_envs, n_agents, channels, 24, 24)
            augmented_obs = torch.stack(augmented_obs, dim=1)

            # Process masks and rewards for both agents
            mask = torch.stack(
                [
                    torch.from_numpy(infos[agent_name][env_idx][4]).bool()
                    for env_idx in range(cfg.n_envs)
                    for agent_name in agent_names
                ]
            ).reshape(cfg.n_envs, n_agents, 24, 24, 4)

            agent_1_rewards = torch.tensor([infos[agent_names[0]][env_idx][5] for env_idx in range(cfg.n_envs)])
            agent_2_rewards = torch.tensor([infos[agent_names[1]][env_idx][5] for env_idx in range(cfg.n_envs)])
            rewards = torch.stack([agent_1_rewards, agent_2_rewards], dim=1)

        return augmented_obs, mask, rewards

    next_obs, infos = envs.reset()
    next_obs, mask, _ = process_observations(next_obs, infos)
    with fabric.device:
        next_done = torch.zeros(cfg.n_envs, dtype=torch.bool)

    for iteration in range(1, cfg.training_iterations + 1):
        for step in range(0, cfg.n_steps):
            global_step += cfg.n_envs
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = mask

            with torch.no_grad():
                # Get actions for player 1 (active player)
                player1_obs = next_obs[:, 0]  # Shape: (n_envs, channels, 24, 24)
                player1_mask = mask[:, 0]  # Shape: (n_envs, 24, 24, 4)
                player1_actions, player1_logprobs, _ = network.get_action(player1_obs, player1_mask)

                # Create dummy actions for player 2
                dummy_actions = torch.ones_like(player1_actions)
                dummy_actions[:, 1:] = 0
                dummy_logprobs = torch.zeros_like(player1_logprobs)

                # Store actions and logprobs with explicit agent dimension
                actions[step, :, 0] = player1_actions
                actions[step, :, 1] = dummy_actions
                logprobs[step, :, 0] = player1_logprobs
                logprobs[step, :, 1] = dummy_logprobs

            # Reshape actions for environment step
            _actions = actions[step].cpu().numpy().astype(int)  # Shape: (n_envs, n_agents, 5)
            next_obs, _, terminations, truncations, infos = envs.step(_actions)
            next_obs, mask, step_rewards = process_observations(next_obs, infos)
            rewards[step] = step_rewards

            # Print rewards if any agent wins
            if torch.any(step_rewards == 1):
                fabric.print(f"Step {step} rewards with 1: {step_rewards}")

            next_done = np.logical_or(terminations, truncations)
            with fabric.device:
                next_done = torch.tensor(next_done, dtype=torch.bool)

        # Calculate returns
        with torch.no_grad(), fabric.device:
            returns = torch.zeros_like(rewards)
            next_value = torch.zeros_like(rewards[0])
            next_non_terminal = 1.0 - next_done.float().unsqueeze(-1)  # Add agent dimension

            for t in reversed(range(cfg.n_steps)):
                returns[t] = rewards[t] + cfg.gamma * next_value * next_non_terminal
                next_value = returns[t]
                next_non_terminal = 1.0 - dones[t].float().unsqueeze(-1)  # Add agent dimension

        # Logging with explicit agent dimension
        total_reward_p1 = rewards[:, :, 0].sum().item()  # First agent
        total_reward_p2 = rewards[:, :, 1].sum().item()  # Second agent

        # Count the number of finished games (each True in dones is one finished game)
        finished_games = dones.sum().item()

        # Compute mean rewards (guard against division by zero)
        mean_reward_p1 = total_reward_p1 / finished_games if finished_games > 0 else 0
        mean_reward_p2 = total_reward_p2 / finished_games if finished_games > 0 else 0

        # Use fabric.print for distributed logging
        fabric.print(
            f"Iteration {iteration} stats - "
            f"Player 1: Total Reward = {total_reward_p1:.2f}, Mean Reward = {mean_reward_p1:.2f}; "
            f"Player 2: Total Reward = {total_reward_p2:.2f}, Mean Reward = {mean_reward_p2:.2f}; "
            f"Finished games: {finished_games}"
        )
        # -------------------------------------------------

        # Use fabric.print instead of print for proper distributed logging
        fabric.print(f"Iteration {iteration} completed")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
