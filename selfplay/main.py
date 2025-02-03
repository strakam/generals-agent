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
    n_obs = 2 * cfg.n_envs
    network = load_network(cfg.checkpoint_path)
    optimizer, _ = network.configure_optimizers(lr=cfg.learning_rate)
    # Setup the network with Fabric
    network, optimizer = fabric.setup(network, optimizer)
    agent = SelfPlayAgent(network=network, batch_size=n_obs // 2, device=fabric.device)

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

    # Add these variables for episode tracking
    episode_rewards = torch.zeros(2 * cfg.n_envs, device=fabric.device)
    episode_lengths = torch.zeros(cfg.n_envs, device=fabric.device)
    num_episodes = 0

    def process_observations(obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with fabric.device:
            # Assume obs comes in with shape (n_envs, 2, channels, 24, 24)
            obs_tensor = torch.from_numpy(obs).half()
            # Split the observation into two parts: one for the active agent and one for the dummy opponent.
            # The shapes: (n_envs, channels, 24, 24)
            active_obs = obs_tensor[:, 0, ...]
            dummy_obs  = obs_tensor[:, 1, ...]
            
            # Process each separately via the agent's augmentation.
            # This ensures that if the augmentation updates internal state (e.g. last_army),
            # you keep the two roles separate.
            augmented_active = agent.augment_observation(active_obs)
            augmented_dummy = agent.augment_observation(dummy_obs)

            # Print sum of channels 10 and 11 separately
            channel_10_sum = augmented_active[:, 10].sum().item()
            channel_11_sum = augmented_active[:, 11].sum().item()
            print(f"Sum of owned cells: {channel_10_sum}")
            print(f"Sum of opponent cells: {channel_11_sum}")

            
            # Reassemble a tensor of shape (n_envs*2, new_channels, 24, 24)
            augmented_obs = torch.stack([augmented_active, augmented_dummy], dim=1)
            augmented_obs = augmented_obs.view(cfg.n_envs * 2, *augmented_active.shape[1:])
            
            # Process the mask as before, ensuring that we use the same ordering: (env0: agent1, agent2, env1: agent1, agent2, …)
            _mask = torch.from_numpy(np.stack([
                infos[agent][env_idx][4]
                for env_idx in range(cfg.n_envs)
                for agent in agent_names
            ]))
            mask = _mask.bool().reshape(cfg.n_envs * 2, 24, 24, 4)
            
            _rewards = torch.tensor([
                infos[agent][env_idx][5]
                for env_idx in range(cfg.n_envs)
                for agent in agent_names
            ])
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
                # Get actions only for first player
                player1_obs = next_obs[::2]  # Take every other observation for player 1
                player1_mask = mask[::2]  # Take every other mask for player 1
                player1_actions, player1_logprobs, _ = network.get_action(player1_obs, player1_mask)

                # Create dummy actions for player 2
                dummy_actions = torch.ones_like(player1_actions)
                dummy_actions[:, 1:] = 0  # Set all but first element to 0
                dummy_logprobs = torch.zeros_like(player1_logprobs)

                # Store actions and logprobs
                actions[step, ::2] = player1_actions
                actions[step, 1::2] = dummy_actions
                logprobs[step, ::2] = player1_logprobs
                logprobs[step, 1::2] = dummy_logprobs

            # Vectorized action assignment
            _actions = (
                actions[step].cpu().numpy().reshape(-1, 2, 5).astype(int)
            )  # Shape: (n_envs, 2 players, 5 action dims)
            print(f"Actions: {_actions}")
            next_obs, _, terminations, truncations, infos = envs.step(_actions)
            next_obs, mask, _rewards = process_observations(next_obs, infos)
            rewards[step] = _rewards
            next_done = np.logical_or(terminations, truncations)
            with fabric.device:
                next_done = torch.tensor(next_done, dtype=torch.bool)

            # Track episode stats
            episode_rewards += _rewards
            episode_lengths += 1

            # Log completed episodes
            if next_done.any():
                for env_idx in range(cfg.n_envs):
                    if next_done[env_idx]:
                        num_episodes += 1
                        # Log metrics for both agents in the finished episode
                        agent1_idx = env_idx * 2
                        agent2_idx = env_idx * 2 + 1

                        if fabric.is_global_zero:
                            # Log episode metrics
                            fabric.log_dict(
                                {
                                    "episode/mean_length": episode_lengths[env_idx].item(),
                                    "episode/agent1_reward": episode_rewards[agent1_idx].item(),
                                    "episode/agent2_reward": episode_rewards[agent2_idx].item(),
                                },
                                step=global_step,
                            )
                            # Print episode metrics
                            fabric.print(
                                f"Episode {num_episodes}: "
                                f"Length={episode_lengths[env_idx].item()}, "
                                f"Agent1 Reward={episode_rewards[agent1_idx].item():.3f}, "
                                f"Agent2 Reward={episode_rewards[agent2_idx].item():.3f}"
                            )

                        # Reset episode tracking for this environment
                        episode_rewards[agent1_idx : agent2_idx + 1] = 0
                        episode_lengths[env_idx] = 0

        # After collecting experience, log training stats
        if fabric.is_global_zero:
            # Get only non-dummy player rewards (every other index starting at 0)
            active_indices = torch.arange(0, rewards.shape[1], 2)
            active_rewards = rewards[:, active_indices]
            active_logprobs = logprobs[:, active_indices]

            fabric.log_dict(
                {
                    "training/mean_returns": active_rewards.mean().item(),
                    "training/mean_entropy": active_logprobs.mean().item(),
                    "training/learning_rate": optimizer.param_groups[0]["lr"],
                    "training/global_step": global_step,
                },
                step=global_step,
            )
            # Print training stats
            fabric.print(
                f"Step {global_step}: "
                f"Returns={active_rewards.mean().item():.3f}, "
                f"Entropy={active_logprobs.mean().item():.3f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

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

        # Take only data from active player (every other index starting at 0)
        # active_indices = torch.arange(0, obs.shape[1], 2)
        # b_obs = obs[:, active_indices].reshape(-1, *obs.shape[2:])
        # b_actions = actions[:, active_indices].reshape(-1, *actions.shape[2:])
        # b_logprobs = logprobs[:, active_indices].reshape(-1)
        # b_returns = returns[:, active_indices].reshape(-1)
        # b_masks = masks[:, active_indices].reshape(-1, *masks.shape[2:])

        # # Store filtered tensors in dictionary for training
        # training_data = {
        #     "observations": b_obs,
        #     "actions": b_actions,
        #     "logprobs": b_logprobs,
        #     "returns": b_returns,
        #     "masks": b_masks,
        # }

        # train(fabric, network, optimizer, training_data, global_step, cfg)

        # Use fabric.print instead of print for proper distributed logging
        fabric.print(f"Iteration {iteration} completed")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
