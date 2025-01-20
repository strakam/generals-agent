import numpy as np
import torch
import lightning as L
from dataclasses import dataclass
from typing import Dict, List, Tuple
from modules.agent import NeuroAgent
from generals import GridFactory, GymnasiumGenerals
from generals.core.action import compute_valid_move_mask
import gymnasium as gym
import argparse
from modules.network import load_network
from modules.agent import SelfPlayAgent

@dataclass
class SelfPlayConfig:
    num_envs: int = 3
    training_iterations: int = 1000
    num_steps: int = 10
    truncation: int = 1500
    checkpoint_path: str = "step=90000.ckpt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
            for _ in range(config.num_envs)
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
    config = SelfPlayConfig()

    # Load agent
    n_obs = 2 * config.num_envs
    network = load_network(config.checkpoint_path)
    agent = SelfPlayAgent(
        network=network,
        batch_size=n_obs,
        device=config.device
    )

    # Create agent names
    agent_names = ["1", "2"]

    # Setup environment
    envs = create_environment(agent_names, config)

    device = config.device
    n_channels = agent.n_channels
    obs = torch.zeros((config.num_steps, 2*config.num_envs, n_channels, 24, 24)).to(device)
    actions = torch.zeros(config.num_steps, 2*config.num_envs, 5).to(device)
    logprobs = torch.zeros(config.num_steps, 2*config.num_envs).to(device)
    rewards = torch.zeros(config.num_steps, config.num_envs).to(device)
    dones = torch.zeros(config.num_steps, config.num_envs).to(device)

    global_step = 0

    def prepare_observations(obs: np.ndarray, infos: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.from_numpy(obs).float().to(device)
        # remove the agent dimension and stack it into the first dimension
        reshaped_obs = obs_tensor.reshape(config.num_envs * 2, -1, 24, 24)
        augmented_obs = agent.augment_observation(reshaped_obs)

        mask = np.stack([info[4] for agent in agent_names for info in infos[agent]])
        mask = torch.from_numpy(mask).to(device)

        return augmented_obs, mask



    next_obs, infos = envs.reset()
    next_obs, mask = prepare_observations(next_obs, infos)
    next_done = torch.zeros(config.num_envs).to(device)

    for iteration in range(1, config.training_iterations + 1):
        for step in range(0, config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            action, combined_logprob = agent.act(next_obs, mask)
            actions[step] = action
            logprobs[step] = combined_logprob

            _actions = actions[step].view(config.num_envs, 2, -1).cpu().numpy().astype(int)
            next_obs, reward, terminations, truncations, infos = envs.step(_actions)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, mask = prepare_observations(next_obs, infos)
            next_done = torch.Tensor(next_done).to(device)
            print(step)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)

