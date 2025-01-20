import torch
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from modules.agent import NeuroAgent, load_agent
from modules.network import Network
from generals import GridFactory, GymnasiumGenerals
import gymnasium as gym
import argparse


@dataclass
class SelfPlayConfig:
    num_envs: int = 1
    training_iterations: int = 1000
    rollout_batch_size: int = 10
    truncation: int = 1500
    checkpoint_path: str = "checkpoints/agent.ckpt"


def create_environment(
    agent_names: List[str], grid_factory: GridFactory, config: SelfPlayConfig
) -> gym.vector.AsyncVectorEnv:
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
    agents: Dict[str, NeuroAgent],
    config: SelfPlayConfig,
) -> List[Tuple]:
    observations, infos = envs.reset()
    dataset = []
    ended = 0
    wins = {agent: 0 for agent in agents.keys()}
    t = 0

    while ended < config.rollout_batch_size:
        actions = {}
        for name, agent in agents.items():
            obs = observations[:, list(agents.keys()).index(name), ...]
            action = agent.act(obs, masks=None)  # Define masks as needed
            actions[name] = action

        # Stack actions appropriately
        actions_stack = torch.stack(
            [actions[name] for name in agents.keys()], dim=1
        ).numpy()
        observations, rewards, terminated, truncated, infos = envs.step(actions_stack)

        # Collect data
        dataset.append((observations, actions, rewards, terminated, truncated))
        t += 1

        done = any(terminated) or any(truncated)
        if done:
            for agent in agents.keys():
                outputs = infos[agent]
                for game in outputs:
                    if game[3] == 1:
                        wins[agent] += 1
                        ended += 1
            print(f"Time {t}, ended {ended}, wins: {wins}")

    return dataset


def train_agents(
    agents: Dict[str, NeuroAgent], dataset: List[Tuple], config: SelfPlayConfig
):
    for name, agent in agents.items():
        # Implement your training logic here
        # Example:
        # agent.train(dataset)
        pass


def main():
    # Initialize hyperparameters
    config = SelfPlayConfig()

    # Load agents
    agents = {
        "agent1": load_agent(
            args.checkpoint1,
            batch_size=args.n_envs,
            eval_mode=True
        ),
        "agent2": load_agent(
            args.checkpoint2 or args.checkpoint1,
            batch_size=args.n_envs,
            eval_mode=True
        ),
    }
    agent_names = list(agents.keys())

    # Setup environment
    gf = GridFactory(
        min_grid_dims=(15, 15),
        max_grid_dims=(24, 24),
    )
    envs = create_environment(agent_names, gf, config)

    for iteration in range(config.training_iterations):
        dataset = execute_rollout(envs, agents, config)
        train_agents(agents, dataset, config)
        print(f"Completed iteration {iteration + 1}/{config.training_iterations}")

    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play Configuration")
    parser.add_argument("--checkpoint1", type=str, required=True, help="Path to the first checkpoint")
    parser.add_argument("--checkpoint2", type=str, help="Path to the second checkpoint")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of environments")
    args = parser.parse_args()
    main()
