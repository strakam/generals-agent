import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from modules.agent import NeuroAgent
from generals import GridFactory, GymnasiumGenerals
import gymnasium as gym
import argparse
from modules.network import load_network
from modules.agent import SelfPlayAgent
import neptune


@dataclass
class SelfPlayConfig:
    n_envs: int = 12
    training_iterations: int = 1000
    n_steps: int = 500
    truncation: int = 1500
    checkpoint_path: str = "step=52000.ckpt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NeptuneLogger:
    """Handles logging experiment metrics to Neptune."""

    def __init__(self, config: SelfPlayConfig):
        self.run = neptune.init_run(
            project="strakam/selfplay",
            name="generals-selfplay",
            tags=["selfplay", "training"],
        )
        self.config = config
        self._log_config()

    def _log_config(self):
        """Log experiment configuration parameters."""
        self.run["parameters"] = {
            "n_envs": self.config.n_envs,
            "training_iterations": self.config.training_iterations,
            "n_steps": self.config.n_steps,
            "truncation": self.config.truncation,
            "checkpoint_path": self.config.checkpoint_path,
            "device": self.config.device,
        }

    def close(self):
        """Close the Neptune run."""
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


def train_agents(
    agents: Dict[str, NeuroAgent], dataset: List[Tuple], config: SelfPlayConfig
):
    for name, agent in agents.items():
        # Implement your training logic here
        # Example:
        # agent.train(dataset)
        pass


def main(args):
    # Initialize hyperparameters
    cfg = SelfPlayConfig()
    logger = NeptuneLogger(cfg)

    # Load agent
    n_obs = 2 * cfg.n_envs
    network = load_network(cfg.checkpoint_path)
    agent = SelfPlayAgent(network=network, batch_size=n_obs, device=cfg.device)

    # Setup environment
    agent_names = ["1", "2"]
    envs = create_environment(agent_names, cfg)

    device = cfg.device
    n_channels = agent.n_channels

    obs_shape = (cfg.n_steps, 2 * cfg.n_envs, n_channels, 24, 24)
    actions_shape = (cfg.n_steps, 2 * cfg.n_envs, 5)
    logprobs_shape = (cfg.n_steps, 2 * cfg.n_envs)
    rewards_shape = (cfg.n_steps, cfg.n_envs)
    dones_shape = (cfg.n_steps, cfg.n_envs)

    obs = torch.zeros(obs_shape, device=device, dtype=torch.float16)
    actions = torch.zeros(actions_shape, device=device, dtype=torch.float16)
    logprobs = torch.zeros(logprobs_shape, device=device)
    rewards = torch.zeros(rewards_shape, device=device, dtype=torch.float16)
    dones = torch.zeros(dones_shape, device=device, dtype=torch.bool)

    global_step = 0

    def prepare_observations(
        obs: np.ndarray, infos: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.from_numpy(obs).half().to(device)  # half() converts to float16
        # remove the agent dimension and stack it into the first dimension
        reshaped_obs = obs_tensor.reshape(cfg.n_envs * 2, -1, 24, 24)
        augmented_obs = agent.augment_observation(reshaped_obs)

        mask = np.stack([info[4] for agent in agent_names for info in infos[agent]])
        mask = torch.from_numpy(mask).bool().to(device)

        return augmented_obs, mask

    next_obs, infos = envs.reset()
    next_obs, mask = prepare_observations(next_obs, infos)
    next_done = torch.zeros(cfg.n_envs, device=device)

    for iteration in range(1, cfg.training_iterations + 1):
        for step in range(0, cfg.n_steps):
            global_step += cfg.n_envs
            obs[step] = next_obs
            dones[step] = next_done

            action, combined_logprob = agent.act(next_obs, mask)
            actions[step] = action
            logprobs[step] = combined_logprob

            _actions = actions[step].view(cfg.n_envs, 2, -1).cpu().numpy().astype(int)
            next_obs, reward, terminations, truncations, infos = envs.step(_actions)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs, mask = prepare_observations(next_obs, infos)
            next_done = torch.tensor(next_done, device=device)

        print(f"Iteration {iteration} completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
