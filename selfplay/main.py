import torch
import lightning as L
from dataclasses import dataclass
from typing import Dict, List, Tuple
from modules.agent import NeuroAgent
from generals import GridFactory, GymnasiumGenerals
import gymnasium as gym
import argparse
from modules.network import load_network


@dataclass
class SelfPlayConfig:
    num_envs: int = 1
    training_iterations: int = 1000
    rollout_batch_size: int = 10
    truncation: int = 1500
    checkpoint_path: str = "step=90000.ckpt"


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


def execute_rollout(
    envs: gym.vector.AsyncVectorEnv,
    network: L.LightningModule,
    agent_names: List[str],
    config: SelfPlayConfig,
) -> List[Tuple]:
    observations, infos = envs.reset()
    dataset = []
    print(observations.shape)
    return dataset


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

    # Load network
    network = load_network(config.checkpoint_path)

    # Create agent names
    agent_names = ["1", "2"]

    # Setup environment
    envs = create_environment(agent_names, config)

    for iteration in range(config.training_iterations):
        dataset = execute_rollout(envs, network, agent_names, config)
        train_agents(agent_names, dataset, config)
        print(f"Completed iteration {iteration + 1}/{config.training_iterations}")

    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    args = parser.parse_args()
    main(args)
