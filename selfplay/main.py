import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
import gymnasium as gym
import argparse
import neptune
from lightning.fabric import Fabric
from tqdm import tqdm

from generals import GridFactory, GymnasiumGenerals
from generals.core.rewards import RewardFn, compute_num_generals_owned
from generals.core.observation import Observation
from generals.core.action import Action
from network import load_network, Network


TORCH_LOGS="recompiles"

class WinLoseRewardFn(RewardFn):
    """A simple reward function. +1 if the agent wins. -1 if they lose."""

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_num_generals_owned = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
        return float(1 * change_in_num_generals_owned)


def generate_random_action(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random actions in format [0, i, j, d, 0] where i,j,d are in range(4)"""
    actions = torch.zeros((batch_size, 5), device=device)
    # Generate random i, j coordinates in range(4)
    actions[:, 1] = torch.randint(0, 4, (batch_size,), device=device)  # i coordinate
    actions[:, 2] = torch.randint(0, 4, (batch_size,), device=device)  # j coordinate
    actions[:, 3] = torch.randint(0, 4, (batch_size,), device=device)  # direction
    # Generate random logprobs (since these are random actions)
    logprobs = torch.ones(batch_size, device=device) * float("-inf")  # -inf logprob for random actions
    return actions, logprobs


@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int = 1000
    n_envs: int = 8
    n_steps: int = 101
    batch_size: int = 64
    n_epochs: int = 2
    truncation: int = 100  # Reduced from 1500 since 4x4 games should be shorter
    grid_size: int = 4  # Already set to 4
    checkpoint_path: str = ""

    # PPO parameters
    gamma: float = 1.0  # Discount factor
    learning_rate: float = 1.5e-4  # Standard PPO learning rate
    max_grad_norm: float = 0.5  # Gradient clipping
    clip_coef: float = 0.2  # PPO clipping coefficient
    ent_coef: float = 0.01  # Increased from 0.00 to encourage exploration

    # Lightning fabric parameters
    strategy: str = "auto"
    precision: str = "32-true"
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
                "precision": self.config.precision,
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
    grid_factory = GridFactory(min_grid_dims=dims, max_grid_dims=dims, general_positions=[(0, 0), (3, 3)])
    return gym.vector.AsyncVectorEnv(
        [
            lambda: GymnasiumGenerals(
                agents=agent_names,
                grid_factory=grid_factory,
                truncation=config.truncation,
                pad_observations_to=24,
                reward_fn=WinLoseRewardFn(),
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
            precision=cfg.precision,
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
            device = self.fabric.device
            self.obs = torch.zeros(self.obs_shape, dtype=torch.float32, device=device)
            self.actions = torch.zeros(self.actions_shape, dtype=torch.float32, device=device)
            self.logprobs = torch.zeros(self.logprobs_shape, dtype=torch.float32, device=device)
            self.rewards = torch.zeros(self.rewards_shape, dtype=torch.float32, device=device)
            self.dones = torch.zeros(self.dones_shape, dtype=torch.bool, device=device)
            self.masks = torch.zeros(self.masks_shape, dtype=torch.bool, device=device)

    def train(self, fabric: Fabric, data: dict):
        """Train the network using PPO on collected rollout data."""
        # Get actual data size
        total_size = data["observations"].shape[0]
        indices = torch.arange(total_size, device=self.fabric.device)

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

        for epoch in range(1, self.cfg.n_epochs + 1):
            # Set epoch for distributed sampler
            if fabric.world_size > 1:
                sampler.set_epoch(epoch)

            pbar = tqdm(batch_sampler, desc=f"Epoch {epoch}/{self.cfg.n_epochs}")

            for batch_idx, batch_indices in enumerate(pbar):
                # Convert indices list to tensor for indexing
                batch_indices_tensor = torch.tensor(batch_indices, device=fabric.device)
                # Index the data using tensor indexing
                batch = {k: v[batch_indices_tensor] for k, v in data.items()}

                loss, pg_loss, entropy_loss, ratio = self.network.training_step(batch, self.cfg)

                # Compute fraction of training samples that were clipped.
                clip_low = 1.0 - self.cfg.clip_coef
                clip_high = 1.0 + self.cfg.clip_coef
                clipped_mask = (ratio < clip_low) | (ratio > clip_high)
                clipfrac = clipped_mask.float().mean()  # fraction in [0,1]

                # Compute approximate KL divergence as the mean value of (ratio - 1 - log(ratio))
                with torch.no_grad():
                    logratio = torch.log(ratio)
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Check if ratios are 1.0 in first epoch and first batch
                if epoch == 1 and batch_idx == 0:
                    assert torch.allclose(
                        ratio, torch.ones_like(ratio), atol=1e-3
                    ), f"Ratios should be 1.0 in first epoch and batch, {ratio}"

                self.optimizer.zero_grad(set_to_none=True)
                fabric.backward(loss)
                grad_norms = self.network.on_after_backward()
                fabric.clip_gradients(self.network, self.optimizer, max_norm=self.cfg.max_grad_norm)
                self.optimizer.step()

                # Log gradient norms
                for module_name, grad_norm in grad_norms.items():
                    self.logger.log_metrics({f"/grad_norm/{module_name}": grad_norm})

                # Log metrics for this batch, including the clipped fraction and approx_kl.
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics(
                    {
                        "train/learning_rate": current_lr,
                        "train/loss": loss.item(),
                        "train/ratio": ratio.mean().item(),
                        "train/policy_loss": pg_loss.mean().item(),
                        "train/entropy_loss": entropy_loss.mean().item(),
                        "train/clipfrac": clipfrac.item(),
                        "train/approx_kl": approx_kl.item(),
                    }
                )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.3f}",
                        "policy_loss": f"{pg_loss.mean().item():.3f}",
                        "entropy_loss": f"{entropy_loss.mean().item():.3f}",
                        "clipfrac": f"{clipfrac.item():.3f}",
                        "approx_kl": f"{approx_kl.item():.3f}",
                    }
                )

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
            obs_tensor = torch.from_numpy(obs).to(self.fabric.device, non_blocking=True)
            augmented_obs = [self.network.augment_observation(obs_tensor[:, idx, ...]) for idx in range(self.n_agents)]
            augmented_obs = torch.stack(augmented_obs, dim=1)

            # Process masks.
            mask = [
                torch.from_numpy(infos[agent_name][env_idx][4]).to(self.fabric.device, non_blocking=True)
                for env_idx in range(self.cfg.n_envs)
                for agent_name in self.agent_names
            ]
            mask = (
                torch.stack(mask)
                .reshape(self.cfg.n_envs, self.n_agents, 24, 24, 4)
                .to(self.fabric.device, non_blocking=True)
            )

            # Process rewards.
            agent_rewards = []
            for agent_name in self.agent_names:
                rewards_agent = torch.tensor(
                    [infos[agent_name][env_idx][5] for env_idx in range(self.cfg.n_envs)],
                    device=self.fabric.device,
                )
                agent_rewards.append(rewards_agent)

            rewards = torch.stack(agent_rewards, dim=1)

            if self.fabric.device.type == "cuda":
                torch.cuda.synchronize()  # Ensure all tensors are ready

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
                    player1_actions, player1_logprobs, _ = self.network(player1_obs, player1_mask)

                    # Generate random actions for the second agent
                    player2_actions, player2_logprobs = generate_random_action(self.cfg.n_envs, self.fabric.device)

                    self.actions[step, :, 0] = player1_actions
                    self.actions[step, :, 1] = player2_actions
                    self.logprobs[step, :, 0] = player1_logprobs
                    self.logprobs[step, :, 1] = player2_logprobs

                # Step the environment.
                _actions = self.actions[step].cpu().numpy().astype(int)
                next_obs, _, terminations, truncations, infos = self.envs.step(_actions)

                next_obs, mask, step_rewards = self.process_observations(next_obs, infos)
                self.rewards[step] = step_rewards
                # Print action-reward pairs for agent 0
                # for env_idx in range(self.cfg.n_envs):
                #     action = _actions[env_idx, 0]  # Get action for agent 0
                #     reward = step_rewards[env_idx, 0].item()  # Get reward for agent 0
                #     self.fabric.print(f"Env {env_idx} - Action: {action}, Reward: {reward}")

                dones = np.logical_or(terminations, truncations)
                next_done = torch.tensor(dones, device=self.fabric.device)

            # Compute returns after collecting rollout.
            with torch.no_grad(), self.fabric.device:
                # Verify rewards are exactly ±1 or 0
                assert torch.all(
                    (torch.abs(self.rewards) == 1.0) | (self.rewards == 0.0)
                ), f"All rewards must be exactly +1, -1, or 0, {self.rewards}"

                returns = torch.zeros_like(self.rewards)
                next_value = torch.zeros_like(self.rewards[0])
                next_non_terminal = 1.0 - next_done.float().unsqueeze(-1)
                for t in reversed(range(self.cfg.n_steps)):
                    returns[t] = self.rewards[t] + self.cfg.gamma * next_value * next_non_terminal
                    next_value = returns[t]
                    next_non_terminal = 1.0 - self.dones[t].float().unsqueeze(-1)

            # Calculate total rewards for each player
            total_reward_p1 = self.rewards[:, :, 0].sum().item()
            total_reward_p2 = self.rewards[:, :, 1].sum().item()

            # Count total number of finished games
            total_games = (torch.abs(self.rewards[:, :, 0]) == 1).sum().item() + 1

            # Calculate winrate (rewards are +1 for win, -1 for loss, 0 for incomplete)
            # So (reward + 1) / 2 converts from [-1,1] to [0,1] range for win rate
            winrate_p1 = ((total_reward_p1 / total_games) + 1) / 2
            winrate_p2 = ((total_reward_p2 / total_games) + 1) / 2

            self.fabric.print(
                f"Iteration {iteration} stats - "
                f"Player 1: Total Reward = {total_reward_p1:.2f}, Winrate = {winrate_p1:.2%}; "
                f"Player 2: Total Reward = {total_reward_p2:.2f}, Winrate = {winrate_p2:.2%}; "
                f"Total games: {total_games}"
            )

            self.logger.log_metrics(
                {
                    "total_reward_p1": total_reward_p1,
                    "winrate_p1": winrate_p1,
                    "total_reward_p2": total_reward_p2,
                    "winrate_p2": winrate_p2,
                    "total_games": total_games,
                }
            )

            # Flatten and prepare dataset for training
            b_obs = self.obs[:, :, 0].reshape(self.cfg.n_steps * self.cfg.n_envs, -1, 24, 24)
            b_actions = self.actions[:, :, 0].reshape(-1, 5)
            b_returns = returns[:, :, 0].reshape(-1)
            b_logprobs = self.logprobs[:, :, 0].reshape(-1)
            b_masks = self.masks[:, :, 0].reshape(-1, 24, 24, 4)

            dataset = {
                "observations": b_obs,
                "actions": b_actions,
                "returns": b_returns,
                "logprobs": b_logprobs,
                "masks": b_masks,
            }

            self.train(self.fabric, dataset)

        self.logger.close()


def main(args):
    cfg = SelfPlayConfig()
    trainer = SelfPlayTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
