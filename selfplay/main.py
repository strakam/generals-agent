import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
import gymnasium as gym
from lightning.fabric import Fabric
from tqdm import tqdm

from generals import GridFactory, GymnasiumGenerals
from network import Network, load_fabric_checkpoint
from rewards import CompositeRewardFn, WinLoseRewardFn
from logger import NeptuneLogger
from model_utils import print_parameter_breakdown

torch.set_float32_matmul_precision("medium")


@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int = 1000
    n_envs: int = 60
    n_steps: int = 8000
    batch_size: int = 800
    n_epochs: int = 4
    truncation: int = 2000
    grid_size: int = 23
    channel_sequence: List[int] = field(default_factory=lambda: [256, 256, 288, 288])
    repeats: List[int] = field(default_factory=lambda: [2, 2, 2, 1])
    checkpoint_path: str = "today.ckpt"
    checkpoint_dir: str = "/root/"

    # PPO parameters
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    learning_rate: float = 1e-5  # Standard PPO learning rate
    max_grad_norm: float = 0.25  # Gradient clipping
    clip_coef: float = 0.2  # PPO clipping coefficient
    ent_coef: float = 0.002  # Increased from 0.00 to encourage exploration
    vf_coef: float = 0.3  # Value function coefficient
    target_kl: float = 0.02  # Target KL divergence
    norm_adv: bool = True  # Whether to normalize advantages
    checkpoint_addition_interval: int = 10

    # Lightning fabric parameters
    strategy: str = "auto"
    precision: str = "32-true"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42


def create_environment(agent_names: List[str], cfg: SelfPlayConfig) -> gym.vector.AsyncVectorEnv:
    # Create environments with different min_generals_distance values
    envs = []
    for i in range(cfg.n_envs):
        # Use min_generals_distance=15 for half of the environments, 20 for the rest
        min_dist = 15 if i < cfg.n_envs // 2 else 20
        print(f"Creating environment with min_generals_distance={min_dist}")
        envs.append(
            lambda min_dist=min_dist: GymnasiumGenerals(
                agents=agent_names,
                grid_factory=GridFactory(mode="generalsio", min_generals_distance=min_dist),
                truncation=cfg.truncation,
                pad_observations_to=24,
                reward_fn=WinLoseRewardFn()
            )
        )

    return gym.vector.AsyncVectorEnv(
        envs,
        shared_memory=True,
    )


class SelfPlayTrainer:
    """Encapsulates setup and training loop for self-play."""

    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg
        # Track self-play iterations
        self.self_play_iteration = 0

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

        opponent_names = ["today"]
        self.opponents = [
            load_fabric_checkpoint(f"{opponent_name}.ckpt", cfg.n_envs) for opponent_name in opponent_names
        ]
        # Set up opponents with Fabric and set to evaluation mode
        for i in range(len(self.opponents)):
            self.opponents[i] = self.fabric.setup(self.opponents[i])
            self.opponents[i].eval()

        # Print detailed parameter breakdown
        if self.fabric.is_global_zero:
            print_parameter_breakdown(self.network)

        self.optimizer, _ = self.network.configure_optimizers(lr=cfg.learning_rate)
        self.network, self.optimizer = self.fabric.setup(self.network, self.optimizer)
        self.fixed_network = self.fabric.setup(self.fixed_network)

        self.network.reset()
        self.fixed_network.reset()

        # Create environment.
        self.agent_names = ["1", "2"]
        self.envs = create_environment(self.agent_names, cfg)

        self.saved_thresholds = set()

        # Setup expected tensor shapes for rollouts.
        self.n_agents = 2
        self.n_channels = self.network.n_channels
        self.obs_shape = (cfg.n_steps, cfg.n_envs, self.n_channels, 24, 24)
        self.masks_shape = (cfg.n_steps, cfg.n_envs, 24, 24, 4)
        self.actions_shape = (cfg.n_steps, cfg.n_envs, 5)
        self.values_shape = (cfg.n_steps, cfg.n_envs)
        self.logprobs_shape = (cfg.n_steps, cfg.n_envs)
        self.rewards_shape = (cfg.n_steps, cfg.n_envs)
        self.dones_shape = (cfg.n_steps, cfg.n_envs)

        self.obs_buffer = torch.empty(
            (cfg.n_envs, self.n_agents, 15, 24, 24),
            dtype=torch.float32,
            pin_memory=torch.cuda.is_available(),
        )
        self.mask_buffer = torch.empty(
            (cfg.n_envs, self.n_agents, 24, 24, 4), dtype=torch.bool, pin_memory=torch.cuda.is_available()
        )

        with self.fabric.device:
            device = self.fabric.device
            self.obs = torch.zeros(self.obs_shape, dtype=torch.float32, device=device)
            self.actions = torch.zeros(self.actions_shape, dtype=torch.float32, device=device)
            self.logprobs = torch.zeros(self.logprobs_shape, dtype=torch.float32, device=device)
            self.values = torch.zeros(self.values_shape, dtype=torch.float32, device=device)
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
            entropies = []
            for batch_idx, batch_indices in enumerate(pbar):
                # Convert indices list to tensor for indexing
                batch_indices_tensor = torch.tensor(batch_indices, device=fabric.device)
                # Index the data using tensor indexing
                batch = {k: v[batch_indices_tensor] for k, v in data.items()}

                loss, pg_loss, value_loss, entropy_loss, ratio, newlogprobs = self.network.training_step(
                    batch, self.cfg
                )
                entropies.append(entropy_loss.mean().item())
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
                        "train/policy_loss": pg_loss.mean().item(),
                        "train/value_loss": value_loss.mean().item(),
                        "train/entropy": entropy_loss.mean().item(),
                        "train/clipfrac": clipfrac.item(),
                        "train/approx_kl": approx_kl.item(),
                    }
                )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.3f}",
                        "policy_loss": f"{pg_loss.mean().item():.3f}",
                        "value_loss": f"{value_loss.mean().item():.3f}",
                        "entropy": f"{entropy_loss.mean().item():.3f}",
                        "clipfrac": f"{clipfrac.item():.3f}",
                        "approx_kl": f"{approx_kl.item():.3f}",
                        "ratio": f"{ratio.mean().item():.3f}",
                    }
                )

            if approx_kl > self.cfg.target_kl:
                break

            if np.mean(entropies) > 1.4:
                self.cfg.ent_coef -= 0.001
            else:
                self.cfg.ent_coef += 0.001
            self.cfg.ent_coef = max(0.0, self.cfg.ent_coef)
            self.fabric.print(f"Changing entropy coefficient: {self.cfg.ent_coef}")

    def process_observations(self, obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes raw observations from the environment.
           This augments memory of the agent, and obtains masks and rewards from the info dict.
        Returns:
            augmented_obs: (n_envs, n_channels, 24, 24) for player 1 only
            mask: (n_envs, n_agents, 24, 24, 4)
            rewards: (n_envs, n_agents)
        """
        with self.fabric.device:
            # Process observations
            self.obs_buffer.copy_(torch.from_numpy(obs))
            self.obs_buffer = self.obs_buffer.to(self.fabric.device, non_blocking=True)

            # Create augmented observations for both players
            augmented_obs = torch.stack(
                [
                    self.network.augment_observation(self.obs_buffer[:, 0, ...]),
                    self.fixed_network.augment_observation(self.obs_buffer[:, 1, ...]),
                ],
                dim=1,
            )

            # Process masks.
            for i, agent_name in enumerate(self.agent_names):
                self.mask_buffer[:, i].copy_(torch.from_numpy(infos[agent_name]["masks"]))
            self.mask_buffer = self.mask_buffer.to(self.fabric.device, non_blocking=True)

            # Process rewards.
            rewards = torch.stack(
                [
                    torch.from_numpy(infos[agent_name]["reward"]).to(self.fabric.device, non_blocking=True)
                    for agent_name in self.agent_names
                ],
                dim=1,
            ).reshape(self.cfg.n_envs, self.n_agents)

        return augmented_obs, self.mask_buffer, rewards

    def run(self):
        """Runs the main training loop."""
        # Pre-allocate action arrays to avoid repeated numpy allocations
        _actions = np.empty((self.cfg.n_envs, 2, 5), dtype=np.int32)

        next_obs, infos = self.envs.reset()
        prev_obs = next_obs
        next_obs, mask, _ = self.process_observations(next_obs, infos)
        next_done = torch.zeros(self.cfg.n_envs, dtype=torch.bool, device=self.fabric.device)

        import random
        self.opponent = self.fixed_network
        for iteration in range(1, self.cfg.training_iterations + 1):
            wins, draws, losses, avg_game_length = 0, 0, 0, 0
            start_time = time.time()
            for step in range(self.cfg.n_steps):
                if step % 750 == 0:
                    # Sample a random opponent from self.opponents every 2000 steps
                    self.opponent = self.opponents[random.randint(0, len(self.opponents) - 1)]

                self.obs[step] = next_obs[:, 0]  # Store player 1's observation directly
                self.dones[step] = next_done
                self.masks[step] = mask[:, 0]  # Only store player 1's mask

                with torch.no_grad(), self.fabric.device:
                    # Get actions for player 1 (learning player)
                    player1_obs = next_obs[:, 0]
                    player1_mask = mask[:, 0]
                    player1_actions, player1_value, player1_logprobs, _ = self.network(player1_obs, player1_mask)
                    _actions[:, 0] = player1_actions.cpu().numpy()

                    # Store player 1's actions, values, and logprobs
                    self.actions[step] = player1_actions
                    self.values[step] = player1_value
                    self.logprobs[step] = player1_logprobs

                    # Get actions for player 2 (fixed network) without storing
                    player2_obs = next_obs[:, 1]
                    player2_mask = mask[:, 1]
                    player2_actions, _ = self.opponent.predict(player2_obs, player2_mask)
                    _actions[:, 1] = player2_actions.cpu().numpy()

                    # Log metrics for player 1
                    probs = torch.exp(player1_logprobs)
                    self.logger.log_metrics(
                        {
                            "step_mean_prob": probs.mean().item(),
                            "step_std_prob": probs.std().item(),
                        }
                    )

                next_obs, _, terminations, truncations, infos = self.envs.step(_actions)
                dones = np.logical_or(terminations, truncations)
                next_done = torch.tensor(dones, device=self.fabric.device)

                # Track game outcomes when episodes finish
                if any(dones):
                    game_times = prev_obs[:, 0, 13, 0, 0]  # Get game times from observations
                    p1_wins = infos[self.agent_names[0]]["winner"]
                    p2_wins = infos[self.agent_names[1]]["winner"]
                    wins += np.sum(dones & p1_wins)
                    losses += np.sum(dones & p2_wins)
                    draws += np.sum(dones & ~p1_wins & ~p2_wins)
                    # Calculate total game time for completed games
                    avg_game_length += np.sum(game_times[dones])
                prev_obs = next_obs

                next_obs, mask, step_rewards = self.process_observations(next_obs, infos)
                self.rewards[step] = step_rewards[:, 0]

            # Compute returns after collecting rollout.
            with torch.no_grad(), self.fabric.device:
                # Get the value estimate for next observation from player 1's network
                player1_next_obs = next_obs[:, 0]
                player1_next_mask = mask[:, 0]
                _, next_value, _, _ = self.network(player1_next_obs, player1_next_mask)
                next_value = next_value.reshape(1, -1)

                # Initialize advantages tensor
                advantages = torch.zeros_like(self.rewards).to(self.fabric.device)
                lastgaelam = 0

                # Calculate advantages using GAE
                for t in reversed(range(self.cfg.n_steps)):
                    if t == self.cfg.n_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1].float()
                        nextvalues = self.values[t + 1]

                    # GAE formula: δ_t = r_t + γV(s_{t+1}) - V(s_t)
                    delta = self.rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - self.values[t]

                    # A_t = δ_t + γλA_{t+1}
                    advantages[t] = lastgaelam = (
                        delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                    )

                # Compute returns as advantage + value (for value function loss)
                returns = advantages + self.values

            # Calculate win/draw/loss percentages
            total_games = wins + draws + losses
            if total_games > 0:
                win_rate = wins / total_games
                draw_rate = draws / total_games
                loss_rate = losses / total_games
                avg_game_length = avg_game_length / total_games  # Calculate average game length
            else:
                win_rate = draw_rate = loss_rate = avg_game_length = 0.0

            self.fabric.print(
                f"Iteration {iteration} stats - "
                f"Player 1: Wins = {win_rate:.2%}, Draws = {draw_rate:.2%}, Losses = {loss_rate:.2%}; "
                f"Avg Game Length = {avg_game_length:.1f}; "
                f"Total games: {total_games}; "
                f"Self-play iteration: {self.self_play_iteration}"
            )

            self.logger.log_metrics(
                {
                    "win_rate": win_rate,
                    "draw_rate": draw_rate,
                    "loss_rate": loss_rate,
                    "avg_game_length": avg_game_length,
                    "total_games": total_games,
                    "self_play_iteration": self.self_play_iteration,
                }
            )
            # Flatten and prepare dataset for training
            b_obs = self.obs.reshape(self.cfg.n_steps * self.cfg.n_envs, -1, 24, 24)
            b_actions = self.actions.reshape(-1, 5)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)
            b_logprobs = self.logprobs.reshape(-1)
            b_masks = self.masks.reshape(-1, 24, 24, 4)

            dataset = {
                "observations": b_obs,
                "actions": b_actions,
                "advantages": b_advantages,
                "returns": b_returns,
                "values": b_values,
                "logprobs": b_logprobs,
                "masks": b_masks,
            }
            gathering_time = time.time() - start_time
            minutes, seconds = divmod(gathering_time, 60)
            print(f"Time taken for gathering: {int(minutes)}m {seconds:.2f}s")

            train_start_time = time.time()
            self.train(self.fabric, dataset)

            training_time = time.time() - train_start_time
            minutes, seconds = divmod(training_time, 60)
            print(f"Time taken for training: {int(minutes)}m {seconds:.2f}s")

            if (iteration+1) % self.cfg.checkpoint_addition_interval == 0:
                self.opponents.append(self.network)
                self.opponents[-1].eval()
                # Save the current network checkpoint
                if self.fabric.is_global_zero:
                    checkpoint_path = f"{self.cfg.checkpoint_dir}cp_{iteration}.ckpt"
                    state = {
                        "model": self.network,
                        "optimizer": self.optimizer,
                    }
                    self.fabric.save(checkpoint_path, state)
                    self.fabric.print(f"Saved checkpoint to {checkpoint_path}")


        self.logger.close()


def main():
    cfg = SelfPlayConfig()
    trainer = SelfPlayTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
