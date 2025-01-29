import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from generals import GridFactory, GymnasiumGenerals
import gymnasium as gym
import argparse
from modules.network import load_network
from modules.agent import SelfPlayAgent
import neptune
from lightning.fabric import Fabric
import pytorch_lightning as L


@dataclass
class SelfPlayConfig:
    n_envs: int = 4
    training_iterations: int = 1000
    n_steps: int = 20
    truncation: int = 1500
    checkpoint_path: str = "step=52000.ckpt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # PPO parameters
    gamma: float = 1.0
    strategy: str = "auto"  # Changed from "ddp" to "auto"
    accelerator: str = "auto"
    devices: int = 1


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
                "device": self.config.device,
                "strategy": self.config.strategy,
                "accelerator": self.config.accelerator,
                "devices": self.config.devices,
            }

    def close(self):
        """Close the Neptune run."""
        if self.fabric.is_global_zero:
            self.run.stop()


def create_environment(
    agent_names: List[str], config: SelfPlayConfig
) -> gym.vector.AsyncVectorEnv:
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


def train(fabric: Fabric, network: L.LightningModule, data: Dict[str, torch.Tensor]):



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

    # Initialize logger after fabric launch
    logger = NeptuneLogger(cfg, fabric)

    # Load agent
    n_obs = 2 * cfg.n_envs
    network = load_network(cfg.checkpoint_path)
    # Setup the network with Fabric
    network = fabric.setup_module(network)
    agent = SelfPlayAgent(network=network, batch_size=n_obs, device=fabric.device)

    # Setup environment
    agent_names = ["1", "2"]
    envs = create_environment(agent_names, cfg)

    n_channels = agent.n_channels

    obs_shape = (cfg.n_steps, 2 * cfg.n_envs, n_channels, 24, 24)
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

    global_step = 0

    def process_observations(
        obs: np.ndarray, infos: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with fabric.device:
            obs_tensor = torch.from_numpy(obs).half()
            # remove the agent dimension and stack it into the first dimension
            reshaped_obs = obs_tensor.reshape(cfg.n_envs * 2, -1, 24, 24)
            augmented_obs = agent.augment_observation(reshaped_obs)

            mask = torch.from_numpy(
                np.stack([info[4] for agent in agent_names for info in infos[agent]])
            ).bool()
            _rewards = torch.tensor(
                [info[5] for agent in agent_names for info in infos[agent]]
            )
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

            action, combined_logprob = agent.act(next_obs, mask)
            actions[step] = action
            logprobs[step] = combined_logprob

            _actions = actions[step].view(cfg.n_envs, 2, -1).cpu().numpy().astype(int)
            next_obs, _, terminations, truncations, infos = envs.step(_actions)
            next_obs, mask, _rewards = process_observations(next_obs, infos)
            rewards[step] = _rewards
            next_done = np.logical_or(terminations, truncations)
            with fabric.device:
                next_done = torch.tensor(next_done, dtype=torch.bool)

        with fabric.device:
            returns = torch.zeros_like(rewards)
            next_value = torch.zeros_like(rewards[0])
            next_non_terminal = torch.repeat_interleave(
                1.0 - next_done.float(), 2, dim=0
            )

        # Calculate returns and advantages
        for t in reversed(range(cfg.n_steps)):
            # Calculate returns with discounting for terminal states
            returns[t] = rewards[t] + cfg.gamma * next_value * next_non_terminal
            next_value = returns[t]
            # Broadcast dones to match the shape of rewards (2 agents per env)
            next_non_terminal = torch.repeat_interleave(
                1.0 - dones[t].float(), 2, dim=0
            )

        # Print shapes of tensors for PPO training
        fabric.print(
            f"Observations shape: {obs.shape}"
        )  # [n_steps, n_envs * 2, obs_dim]
        fabric.print(
            f"Actions shape: {actions.shape}"
        )  # [n_steps, n_envs * 2, action_dim]
        fabric.print(f"Logprobs shape: {logprobs.shape}")  # [n_steps, n_envs * 2]
        fabric.print(f"Returns shape: {returns.shape}")  # [n_steps, n_envs * 2]

        # Flatten data across first two axes (n_steps, n_envs * 2) -> (n_steps * n_envs * 2, ...)
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_returns = returns.reshape(-1)

        # Store flattened tensors in dictionary for training
        training_data = {
            "observations": b_obs,
            "actions": b_actions,
            "logprobs": b_logprobs,
            "returns": b_returns,
        }

        train(fabric, network, training_data)

        # Use fabric.print instead of print for proper distributed logging
        fabric.print(f"Iteration {iteration} completed")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
