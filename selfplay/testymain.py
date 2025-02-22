import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
import gymnasium as gym
import argparse
import neptune
from lightning.fabric import Fabric
from tqdm import tqdm
import os

from generals import GridFactory, GymnasiumGenerals
from generals.core.rewards import RewardFn, compute_num_generals_owned
from generals.core.observation import Observation
from generals.core.action import Action
from network import load_fabric_checkpoint, Network

torch.set_float32_matmul_precision("medium")


class WinLoseRewardFn(RewardFn):
    """A simple reward function. +1 if the agent wins. -1 if they lose."""

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_num_generals_owned = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
        return float(1 * change_in_num_generals_owned)

class GainLandRewardFn(RewardFn):
    """A simple reward function. +1 if the agent gains land. -1 if they lose land."""

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_land_count = obs.owned_land_count - prior_obs.owned_land_count
        return float(1 * change_in_land_count)


@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int = 1000
    n_envs: int = 256
    n_steps: int = 400
    batch_size: int = 256
    n_epochs: int = 4
    truncation: int = 400 # Reduced from 1500 since 4x4 games should be shorter
    grid_size: int = 23  # Already set to 4
    channel_sequence: List[int] = field(default_factory=lambda: [32, 48, 64, 64])
    repeats: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    checkpoint_path: str = ""
    # checkpoint_path: str = "checkpoints/win_rate_0.00.ckpt"
    checkpoint_dir: str = "/storage/praha1/home/strakam3/selfplay_checkpoints/"
    # checkpoint_dir: str = "checkpoints/"

    # Win rate thresholds for checkpointing (15%, 30%, 45%, etc.)
    win_rate_thresholds: List[float] = field(default_factory=lambda: [0.15, 0.30, 0.45, 0.60, 0.75, 0.90])

    # PPO parameters
    gamma: float = 0.95  # Discount factor
    learning_rate: float = 1.0e-4  # Standard PPO learning rate
    max_grad_norm: float = 0.25  # Gradient clipping
    clip_coef: float = 0.2  # PPO clipping coefficient
    ent_coef: float = 0.01  # Increased from 0.00 to encourage exploration
    target_kl: float = 0.02  # Target KL divergence

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
                self.run[f"{key}"].log(value)

    def close(self):
        """Close the Neptune run."""
        if self.fabric.is_global_zero:
            self.run.stop()


def create_environment(agent_names: List[str], cfg: SelfPlayConfig) -> gym.vector.AsyncVectorEnv:
    # grid_factory = GridFactory(mode="generalsio")
    min_grid_size = 24
    grid_factory = GridFactory(min_grid_size=min_grid_size, max_grid_size=min_grid_size, mountain_density=0.2)
    return gym.vector.AsyncVectorEnv(
        [
            lambda: GymnasiumGenerals(
                agents=agent_names,
                grid_factory=grid_factory,
                truncation=cfg.truncation,
                pad_observations_to=24,
                reward_fn=GainLandRewardFn(),
            )
            for _ in range(cfg.n_envs)
        ],
        shared_memory=True,
    )


class SelfPlayTrainer:
    """Encapsulates setup and training loop for self-play."""

    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg
        # Track which win rate thresholds we've already saved checkpoints for
        self.saved_thresholds = set()

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

        # Load or initialize the learning network
        if cfg.checkpoint_path != "":
            self.network = load_fabric_checkpoint(cfg.checkpoint_path, cfg.n_envs)
            # Load the fixed network from the same checkpoint
            self.fixed_network = load_fabric_checkpoint(cfg.checkpoint_path, cfg.n_envs)
            self.fixed_network.eval()  # Set fixed network to evaluation mode
        else:
            seq = cfg.channel_sequence
            self.network = Network(batch_size=cfg.n_envs, channel_sequence=seq, repeats=cfg.repeats)
            self.fixed_network = Network(batch_size=cfg.n_envs, channel_sequence=seq, repeats=cfg.repeats)
            self.fixed_network.eval()
            n_params = sum(p.numel() for p in self.network.parameters())
            print(f"Number of parameters: {n_params:,}")

        self.optimizer, _ = self.network.configure_optimizers(lr=cfg.learning_rate)
        self.network, self.optimizer = self.fabric.setup(self.network, self.optimizer)
        self.fixed_network = self.fabric.setup(self.fixed_network)

        self.network.reset()
        self.fixed_network.reset()

        # Create environment.
        self.agent_names = ["1", "2"]
        self.envs = create_environment(self.agent_names, cfg)

        # Setup expected tensor shapes for rollouts.
        self.n_agents = 2
        self.n_channels = self.network.n_channels
        self.obs_shape = (cfg.n_steps, cfg.n_envs, self.n_channels, 24, 24)
        self.masks_shape = (cfg.n_steps, cfg.n_envs, 24, 24, 4)
        self.actions_shape = (cfg.n_steps, cfg.n_envs, 5)
        self.logprobs_shape = (cfg.n_steps, cfg.n_envs)
        self.rewards_shape = (cfg.n_steps, cfg.n_envs)
        self.dones_shape = (cfg.n_steps, cfg.n_envs)

        with self.fabric.device:
            device = self.fabric.device
            self.obs = torch.zeros(self.obs_shape, dtype=torch.float32, device=device)
            self.actions = torch.zeros(self.actions_shape, dtype=torch.float32, device=device)
            self.logprobs = torch.zeros(self.logprobs_shape, dtype=torch.float32, device=device)
            self.rewards = torch.zeros(self.rewards_shape, dtype=torch.float32, device=device)
            self.dones = torch.zeros(self.dones_shape, dtype=torch.bool, device=device)
            self.masks = torch.zeros(self.masks_shape, dtype=torch.bool, device=device)

    def save_checkpoint(self, win_rate: float, threshold: float):
        """Save a checkpoint in Fabric format when crossing a win rate threshold."""
        if self.fabric.is_global_zero:  # Only save on main process
            checkpoint_name = f"winrate_{threshold:.2f}.ckpt"
            checkpoint_path = f"{self.cfg.checkpoint_dir}/{checkpoint_name}"

            # Create checkpoint directory if it doesn't exist
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

            # Create state dictionary with everything needed to resume training
            state = {
                "model": self.network,
                "optimizer": self.optimizer,
            }

            # Let Fabric handle the saving
            self.fabric.save(checkpoint_path, state)
            print(f"Saved Fabric checkpoint at win rate {win_rate:.2%} to {checkpoint_path}")

            # Log to Neptune
            self.logger.log_metrics({"checkpoint/fabric_win_rate": win_rate})

    def check_and_save_checkpoints(self, win_rate: float):
        """Check if we've crossed any win rate thresholds and save checkpoints if needed."""
        for threshold in self.cfg.win_rate_thresholds:
            if win_rate >= threshold and threshold not in self.saved_thresholds:
                # Save both types of checkpoints
                self.save_checkpoint(win_rate, threshold)
                self.saved_thresholds.add(threshold)

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

                loss, pg_loss, entropy_loss, ratio, newlogprobs = self.network.training_step(batch, self.cfg)
                oldlogprobs = batch["logprobs"]

                # Compute approximate KL divergence as the mean value of (ratio - 1 - log(ratio))
                with torch.no_grad():
                    logratio = torch.log(ratio)
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()  # fraction in [0,1]

                self.optimizer.zero_grad(set_to_none=True)
                fabric.backward(loss)

                # Unscale the gradients before logging or clipping if using mixed precision.
                if hasattr(fabric, "scaler") and fabric.scaler is not None:
                    fabric.scaler.unscale_(self.optimizer)

                # Now the gradients are in their true scale, ensuring that logging and clipping are applied correctly.
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
                        "train/logratio": logratio.mean().item(),
                        "train/policy_loss": pg_loss.mean().item(),
                        "train/entropy": entropy_loss.mean().item(),
                        "train/clipfrac": clipfrac.item(),
                        "train/approx_kl": approx_kl.item(),
                        "train/newlogprobs": newlogprobs.mean().item(),
                        "train/oldlogprobs": oldlogprobs.mean().item(),
                    }
                )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.3f}",
                        "policy_loss": f"{pg_loss.mean().item():.3f}",
                        "entropy": f"{entropy_loss.mean().item():.3f}",
                        "clipfrac": f"{clipfrac.item():.3f}",
                        "approx_kl": f"{approx_kl.item():.3f}",
                        "logratio": f"{logratio.mean().item():.3f}",
                        "ratio": f"{ratio.mean().item():.3f}",
                        "min_ratio": f"{ratio.min().item():.3f}",
                        "newlogprobs": f"{newlogprobs.mean().item():.3f}",
                        "oldlogprobs": f"{oldlogprobs.mean().item():.3f}",
                    }
                )

            if approx_kl > self.cfg.target_kl:
                break

    def process_observations(self, obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes raw observations from the environment.
           This augments memory of the agent, and obtains masks and rewards from the info dict.
        Returns:
            augmented_obs: (n_envs, n_channels, 24, 24) for player 1 only
            mask: (n_envs, n_agents, 24, 24, 4)
            rewards: (n_envs, n_agents)
        """
        with self.fabric.device:
            # Process observations - only for player 1
            obs_tensor = torch.from_numpy(obs).to(self.fabric.device, non_blocking=True)
            agent1_augmented_obs = self.network.augment_observation(obs_tensor[:, 0, ...])
            agent2_augmented_obs = self.fixed_network.augment_observation(obs_tensor[:, 1, ...])
            augmented_obs = torch.stack([agent1_augmented_obs, agent2_augmented_obs], dim=1)

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
            wins, draws, losses = 0, 0, 0
            land1, land2 = 0, 0

            for step in range(0, self.cfg.n_steps):
                global_step += self.cfg.n_envs
                self.obs[step] = next_obs[:, 0]  # Store player 1's observation directly
                self.dones[step] = next_done
                self.masks[step] = mask[:, 0]  # Only store player 1's mask

                with torch.no_grad():
                    # Get actions for player 1 (learning player)
                    player1_obs = next_obs[:, 0]
                    player1_mask = mask[:, 0]
                    player1_actions, player1_logprobs, _ = self.network(player1_obs, player1_mask)

                    # Store player 1's actions and logprobs
                    self.actions[step] = player1_actions
                    self.logprobs[step] = player1_logprobs

                    # Get actions for player 2 (fixed player) without storing
                    player2_obs = next_obs[:, 1]
                    player2_mask = mask[:, 1]
                    player2_actions, _, _ = self.fixed_network(player2_obs, player2_mask)

                    # Log metrics for player 1
                    probs = torch.exp(player1_logprobs)
                    mean_prob = probs.mean().item()
                    std_prob = probs.std().item()
                    self.logger.log_metrics(
                        {
                            "step_mean_prob": mean_prob,
                            "step_std_prob": std_prob,
                        }
                    )

                # Combine actions for environment step
                _actions = torch.stack([player1_actions, player2_actions], dim=1).cpu().numpy().astype(int)
                _prev_obs = next_obs.clone()
                next_obs, _, terminations, truncations, infos = self.envs.step(_actions)

                next_obs, mask, step_rewards = self.process_observations(next_obs, infos)

                # Store only player 1's rewards
                self.rewards[step] = step_rewards[:, 0]

                dones = np.logical_or(terminations, truncations)
                next_done = torch.tensor(dones, device=self.fabric.device)

                # Track game outcomes when episodes finish
                if any(dones):
                    for env_idx in range(self.cfg.n_envs):
                        if dones[env_idx]:
                            land1 += _prev_obs[env_idx, 0, 10].sum().item()
                            land2 += _prev_obs[env_idx, 1, 10].sum().item()
                            p1_won = infos[self.agent_names[0]][env_idx][3]
                            p2_won = infos[self.agent_names[1]][env_idx][3]
                            if p1_won:
                                wins += 1
                            elif p2_won:
                                losses += 1
                            else:
                                draws += 1

            # Compute returns after collecting rollout.
            with torch.no_grad(), self.fabric.device:
                returns = torch.zeros_like(self.rewards)
                next_value = torch.zeros_like(self.rewards[0])
                next_non_terminal = 1.0 - next_done.float()
                for t in reversed(range(self.cfg.n_steps)):
                    returns[t] = self.rewards[t] + self.cfg.gamma * next_value * next_non_terminal
                    next_value = returns[t]
                    next_non_terminal = 1.0 - self.dones[t].float()

            # Calculate win/draw/loss percentages
            total_games = wins + draws + losses
            if total_games > 0:
                win_rate = wins / total_games
                draw_rate = draws / total_games
                loss_rate = losses / total_games

                # Check if we should save a checkpoint
                self.check_and_save_checkpoints(win_rate)
            else:
                win_rate = draw_rate = loss_rate = 0.0

            self.fabric.print(
                f"Iteration {iteration} stats - "
                f"Player 1: Wins = {win_rate:.2%}, Draws = {draw_rate:.2%}, Losses = {loss_rate:.2%}; "
                f"Total games: {total_games}"
            )

            self.logger.log_metrics(
                {
                    "win_rate": win_rate,
                    "draw_rate": draw_rate,
                    "loss_rate": loss_rate,
                    "total_games": total_games,
                    "land1": land1 / total_games,
                    "land2": land2 / total_games,
                }
            )

            # Flatten and prepare dataset for training
            b_obs = self.obs.reshape(self.cfg.n_steps * self.cfg.n_envs, -1, 24, 24)
            b_actions = self.actions.reshape(-1, 5)
            b_returns = returns.reshape(-1)
            b_logprobs = self.logprobs.reshape(-1)
            b_masks = self.masks.reshape(-1, 24, 24, 4)

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


def clean_checkpoint(checkpoint_path, output_path=None):
    """
    Loads a checkpoint, removes the config object and threshold-related fields, and re-saves it.

    Args:
        checkpoint_path (str): Path to the original checkpoint file.
        output_path (str, optional): Path to save the cleaned checkpoint.
                                     If not provided, will overwrite the original.
    """
    # Load the checkpoint from the specified file.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Remove the problematic fields if they exist
    fields_to_remove = ["config", "threshold", "saved_thresholds"]
    for field in fields_to_remove:
        if field in checkpoint:
            print(f"Removing '{field}' from checkpoint.")
            del checkpoint[field]
        else:
            print(f"No '{field}' key found in checkpoint.")

    # Determine the output path: either overwrite or save to a new file.
    if output_path is None:
        output_path = checkpoint_path

    # Save the cleaned checkpoint.
    torch.save(checkpoint, output_path)
    print(f"Checkpoint successfully saved to '{output_path}' without the removed fields.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
