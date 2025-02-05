import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
import gymnasium as gym
import argparse
import neptune
from lightning.fabric import Fabric

from generals import GridFactory, GymnasiumGenerals
from generals.core.rewards import LandRewardFn
from network import load_network, Network


@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int = 1000
    n_envs: int = 256
    n_steps: int = 100
    batch_size: int = 512
    n_epochs: int = 3
    truncation: int = 1500
    grid_size: int = 8
    checkpoint_path: str = ""#"step=52000.ckpt"

    # PPO parameters
    gamma: float = 0.99 # Discount factor
    learning_rate: float = 3e-4  # Standard PPO learning rate
    max_grad_norm: float = 0.5  # Gradient clipping
    clip_coef: float = 0.2  # PPO clipping coefficient
    ent_coef: float = 0.00  # Entropy coefficient

    # Lightning fabric parameters
    strategy: str = "auto"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42


class NeptuneLogger:
    """Handles logging experiment metrics to Neptune."""

    def __init__(self, config: SelfPlayConfig, fabric: Fabric):
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
                "devices": self.config.devices,
            }

    def log_metrics(self, metrics: dict):
        """Logs a batch of metrics to Neptune."""
        if self.fabric.is_global_zero:
            for key, value in metrics.items():
                self.run[f"metrics/{key}"].log(value)

    def close(self):
        """Close the Neptune run."""
        if self.fabric.is_global_zero:
            self.run.stop()


def create_environment(agent_names: List[str], config: SelfPlayConfig) -> gym.vector.AsyncVectorEnv:
    dims = (config.grid_size, config.grid_size)
    grid_factory = GridFactory(min_grid_dims=dims, max_grid_dims=dims)
    return gym.vector.AsyncVectorEnv(
        [
            lambda: GymnasiumGenerals(
                agents=agent_names,
                grid_factory=grid_factory,
                truncation=config.truncation,
                pad_observations_to=24,
                reward_fn=LandRewardFn(),
            )
            for _ in range(config.n_envs)
        ],
    )


class SelfPlayTrainer:
    """Encapsulates setup and training loop for self-play."""

    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg

        # Initialize Fabric and seed the experiment.
        self.fabric = Fabric(
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            strategy=cfg.strategy,
        )
        self.fabric.launch()
        self.fabric.seed_everything(cfg.seed)

        # Set up logger.
        self.logger = NeptuneLogger(cfg, self.fabric)

        # Load network or initialize new one, set up optimizer, and configure the agent.
        if cfg.checkpoint_path != "":
            self.network = load_network(cfg.checkpoint_path, cfg.n_envs)
        else:
            self.network = Network(batch_size=cfg.n_envs)
        self.optimizer, _ = self.network.configure_optimizers(lr=cfg.learning_rate)
        self.network, self.optimizer = self.fabric.setup(self.network, self.optimizer)

        self.network.reset()

        # Create environment.
        agent_names = ["1", "2"]
        self.envs = create_environment(agent_names, cfg)
        self.agent_names = agent_names

        # Setup expected tensor shapes for rollouts.
        self.n_agents = 2
        self.n_channels = self.network.n_channels
        self.obs_shape = (cfg.n_steps, cfg.n_envs, self.n_agents, self.n_channels, 24, 24)
        self.masks_shape = (cfg.n_steps, cfg.n_envs, self.n_agents, 24, 24, 4)
        self.actions_shape = (cfg.n_steps, cfg.n_envs, self.n_agents, 5)
        self.logprobs_shape = (cfg.n_steps, cfg.n_envs, self.n_agents)
        self.rewards_shape = (cfg.n_steps, cfg.n_envs, self.n_agents)
        self.dones_shape = (cfg.n_steps, cfg.n_envs)

        with self.fabric.device:
            self.obs = torch.zeros(self.obs_shape, dtype=torch.float16)
            self.actions = torch.zeros(self.actions_shape, dtype=torch.float16)
            self.logprobs = torch.zeros(self.logprobs_shape)
            self.rewards = torch.zeros(self.rewards_shape, dtype=torch.float16)
            self.dones = torch.zeros(self.dones_shape, dtype=torch.bool)
            self.masks = torch.zeros(self.masks_shape, dtype=torch.bool)

    def train(self, fabric: Fabric, data: dict):
        """Train the network using PPO on collected rollout data."""
        # Get actual data size
        total_size = data["observations"].shape[0]
        indices = torch.arange(total_size)

        effective_batch_size = min(self.cfg.batch_size, total_size)

        # Choose sampler based on distributed setting
        if fabric.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                indices,
                num_replicas=fabric.world_size,
                rank=fabric.global_rank,
                shuffle=True,
                seed=self.cfg.seed,
            )
        else:
            sampler = torch.utils.data.RandomSampler(indices)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=effective_batch_size,
            drop_last=False,
        )

        self.network.train()
        for epoch in range(self.cfg.n_epochs):
            # Set epoch for distributed sampler
            if fabric.world_size > 1:
                sampler.set_epoch(epoch)

            for batch_indices in batch_sampler:
                # Convert indices list to tensor for indexing
                batch_indices_tensor = torch.tensor(batch_indices, device=fabric.device)
                # Index the data using tensor indexing
                batch = {k: v[batch_indices_tensor] for k, v in data.items()}
                loss = self.network.ppo_loss(batch, self.cfg)

                self.optimizer.zero_grad(set_to_none=True)
                fabric.backward(loss)
                fabric.clip_gradients(self.network, self.optimizer, max_norm=self.cfg.max_grad_norm)
                self.optimizer.step()
                # print loss
                fabric.print(f"Loss: {loss.item()}")

    def process_observations(self, obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes raw observations from the environment.
           This augments memory of the agent, and obtains masks and rewards from the info dict.
        Returns:
            augmented_obs: (n_envs, n_agents, channels, 24, 24)
            mask: (n_envs, n_agents, 24, 24, 4)
            rewards: (n_envs, n_agents)
        """
        with self.fabric.device:
            # Assume obs shape: (n_envs, 2, channels, 24, 24)
            obs_tensor = torch.from_numpy(obs).to(self.fabric.device)
            augmented_obs = [self.network.augment_observation(obs_tensor[:, idx, ...]) for idx in range(self.n_agents)]
            augmented_obs = torch.stack(augmented_obs, dim=1)

            # Process masks.
            mask = [
                torch.from_numpy(infos[agent_name][env_idx][4]).bool()
                for env_idx in range(self.cfg.n_envs)
                for agent_name in self.agent_names
            ]
            mask = torch.stack(mask).reshape(self.cfg.n_envs, self.n_agents, 24, 24, 4).to(self.fabric.device)

            # Process rewards.
            agent_rewards = []
            for agent_name in self.agent_names:
                rewards_agent = torch.tensor([infos[agent_name][env_idx][5] for env_idx in range(self.cfg.n_envs)])
                agent_rewards.append(rewards_agent)

            rewards = torch.stack(agent_rewards, dim=1).to(self.fabric.device)

        return augmented_obs, mask, rewards

    def run(self):
        """Runs the main training loop."""

        global_step = 0

        for iteration in range(1, self.cfg.training_iterations + 1):
            next_obs, infos = self.envs.reset()
            next_obs, mask, _ = self.process_observations(next_obs, infos)
            next_done = torch.zeros(self.cfg.n_envs, dtype=torch.bool, device=self.fabric.device)
            for step in range(0, self.cfg.n_steps):
                global_step += self.cfg.n_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                self.masks[step] = mask

                with torch.no_grad():
                    # Compute actions for the active player.
                    player1_obs = next_obs[:, 0]
                    player1_mask = mask[:, 0]
                    player1_actions, player1_logprobs, _ = self.network.get_action(player1_obs, player1_mask)

                    # Create dummy actions for the second agent.
                    dummy_logprobs = torch.zeros_like(player1_logprobs)
                    dummy_actions = torch.zeros_like(player1_actions)
                    dummy_actions[:, 0] = 1  # pass action

                    self.actions[step, :, 0] = player1_actions
                    self.actions[step, :, 1] = dummy_actions
                    self.logprobs[step, :, 0] = player1_logprobs
                    self.logprobs[step, :, 1] = dummy_logprobs

                # Step the environment.
                _actions = self.actions[step].cpu().numpy().astype(int)
                next_obs, _, terminations, truncations, infos = self.envs.step(_actions)
                next_obs, mask, step_rewards = self.process_observations(next_obs, infos)
                self.rewards[step] = step_rewards

                dones = np.logical_or(terminations, truncations)
                next_done = torch.tensor(dones, device=self.fabric.device)

            # Compute returns after collecting rollout.
            with torch.no_grad(), self.fabric.device:
                # Verify rewards are exactly ±1 or 0
                # assert torch.all(
                #     (torch.abs(self.rewards) == 1.0) | (self.rewards == 0.0)
                # ), f"All rewards must be exactly +1, -1, or 0, {self.rewards}"

                returns = torch.zeros_like(self.rewards)
                next_value = torch.zeros_like(self.rewards[0])
                next_non_terminal = 1.0 - next_done.float().unsqueeze(-1)
                for t in reversed(range(self.cfg.n_steps)):
                    returns[t] = self.rewards[t] + self.cfg.gamma * next_value * next_non_terminal
                    next_value = returns[t]
                    next_non_terminal = 1.0 - self.dones[t].float().unsqueeze(-1)

            # Flatten and prepare dataset for training
            b_obs = self.obs[:, :, 0].reshape(self.cfg.n_steps * self.cfg.n_envs, -1, 24, 24)
            b_actions = self.actions[:, :, 0].reshape(self.cfg.n_steps * self.cfg.n_envs, -1)
            b_returns = returns[:, :, 0].reshape(self.cfg.n_steps * self.cfg.n_envs, -1)
            b_logprobs = self.logprobs[:, :, 0].reshape(self.cfg.n_steps * self.cfg.n_envs)
            b_masks = self.masks[:, :, 0].reshape(self.cfg.n_steps * self.cfg.n_envs, 24, 24, 4)

            dataset = {
                "observations": b_obs,
                "actions": b_actions,
                "returns": b_returns,
                "logprobs": b_logprobs,
                "masks": b_masks,
            }

            self.train(self.fabric, dataset)

            total_reward_p1 = self.rewards[:, :, 0].sum().item()
            total_reward_p2 = self.rewards[:, :, 1].sum().item()
            finished_games = self.dones.sum().item()
            # mean_reward_p1 = total_reward_p1 / finished_games if finished_games > 0 else 0
            # mean_reward_p2 = total_reward_p2 / finished_games if finished_games > 0 else 0
            mean_reward_p1 = total_reward_p1 / self.cfg.n_envs
            mean_reward_p2 = total_reward_p2 / self.cfg.n_envs

            self.fabric.print(
                f"Iteration {iteration} stats - "
                f"Player 1: Total Reward = {total_reward_p1:.2f}, Mean Reward = {mean_reward_p1:.2f}; "
                f"Player 2: Total Reward = {total_reward_p2:.2f}, Mean Reward = {mean_reward_p2:.2f}; "
                f"Finished games: {finished_games}"
            )
            self.fabric.print(f"Iteration {iteration} completed")

            self.logger.log_metrics(
                {
                    "total_reward_p1": total_reward_p1,
                    "mean_reward_p1": mean_reward_p1,
                    "total_reward_p2": total_reward_p2,
                    "mean_reward_p2": mean_reward_p2,
                    "finished_games": finished_games,
                }
            )

        self.logger.close()


def main(args):
    cfg = SelfPlayConfig()
    trainer = SelfPlayTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
