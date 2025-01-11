import gymnasium as gym
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from torch import nn

from generals import GridFactory
from generals.envs import GymnasiumGenerals
from supervised.network import Network
from supervised.neuro_agent import NeuroAgent


@dataclass
class ExperimentConfig:
    """Configuration parameters for the experiment."""

    n_envs: int = 5
    num_games: int = 5
    checkpoint_dir: str = "checkpoints/sup145"
    min_grid_size: int = 10
    max_grid_size: int = 15
    mountain_density: float = 0.08
    city_density: float = 0.05
    truncation_steps: int = 1500
    grid_padding: int = 24
    channel_sequence: List[int] = (256, 320, 384, 384)


class EnvironmentFactory:
    """Creates and manages game environments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def create_grid_factory(self) -> GridFactory:
        return GridFactory(
            min_grid_dims=(self.config.min_grid_size, self.config.min_grid_size),
            max_grid_dims=(self.config.max_grid_size, self.config.max_grid_size),
            mountain_density=self.config.mountain_density,
            city_density=self.config.city_density,
        )

    def create_environment(self, agent_names: List[str]) -> GymnasiumGenerals:
        grid_factory = self.create_grid_factory()
        return GymnasiumGenerals(
            agent_ids=agent_names,
            grid_factory=grid_factory,
            truncation=self.config.truncation_steps,
            pad_to=self.config.grid_padding,
        )


class AgentLoader:
    """Handles loading and initialization of agents."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_agent(self, checkpoint_path: str) -> NeuroAgent:
        network = torch.load(checkpoint_path, map_location=self.device)
        model = self._initialize_model(network["state_dict"])
        agent_id = Path(checkpoint_path).stem.split("=")[-1]

        return NeuroAgent(
            model, id=agent_id, batch_size=self.config.n_envs, device=self.device
        )

    def _initialize_model(self, state_dict: Dict) -> nn.Module:
        model = Network(channel_sequence=self.config.channel_sequence, compile=True)
        model_keys = model.state_dict().keys()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered_state_dict)
        model.eval()
        return model


class AgentEvaluator:
    """Handles agent matchups and evaluations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.env_factory = EnvironmentFactory(config)

    def run_matchup(self, agent1: NeuroAgent, agent2: NeuroAgent) -> Dict[str, int]:
        names = [agent1.id, agent2.id]
        env_creator = lambda: self.env_factory.create_environment(names)
        envs = gym.vector.AsyncVectorEnv(
            [env_creator for _ in range(self.config.n_envs)]
        )

        return self._play_games(envs, agent1, agent2)

    def _play_games(
        self, envs, agent1: NeuroAgent, agent2: NeuroAgent
    ) -> Dict[str, int]:
        observations, infos = envs.reset()
        wins = {agent1.id: 0, agent2.id: 0}
        ended_games = 0

        while ended_games < self.config.num_games:
            observations, terminated, truncated, infos = self._play_step(
                envs, observations, infos, agent1, agent2
            )

            if any(terminated):
                ended_games += self._process_game_results(infos, wins)

        return wins

    def _play_step(
        self, envs, observations, infos, agent1, agent2
    ) -> Tuple[np.ndarray, bool, bool, Dict]:
        agent_1_obs, agent_2_obs = observations[:, 0, ...], observations[:, 1, ...]
        masks = [
            np.stack([info[-1] for info in infos[agent.id]])
            for agent in (agent1, agent2)
        ]

        actions = np.stack(
            [agent1.act(agent_1_obs, masks[0]), agent2.act(agent_2_obs, masks[1])],
            axis=1,
        )

        observations, _, terminated, truncated, infos = envs.step(actions)
        return observations, terminated, truncated, infos

    def _process_game_results(self, infos: Dict, wins: Dict[str, int]) -> int:
        ended = 0
        for agent_id in wins.keys():
            for game in infos[agent_id]:
                if game[3] == 1:  # Victory condition
                    wins[agent_id] += 1
                    ended += 1
        return ended


class ResultVisualizer:
    """Handles visualization of tournament results."""

    def create_heatmap(
        self,
        winrates: List[List[float]],
        agent_names: List[str],
        output_path: str = "/storage/praha1/home/strakam3/winrates.png",
    ):
        plt.figure(figsize=(8, 6))
        mask = np.eye(len(agent_names), dtype=bool)

        sns.heatmap(
            winrates,
            annot=True,
            fmt=".2f",
            cmap=sns.color_palette("RdYlGn", as_cmap=True),
            mask=mask,
            cbar_kws={"label": "Win Rate"},
        )

        self._configure_plot(agent_names)
        plt.savefig(output_path)
        # plt.show()

    def _configure_plot(self, agent_names: List[str]):
        plt.xticks(
            ticks=np.arange(len(agent_names)) + 0.5,
            labels=[name.split("=")[-1] for name in agent_names],
            rotation=45,
        )
        plt.yticks(
            ticks=np.arange(len(agent_names)) + 0.5,
            labels=[name.split("=")[-1] for name in agent_names],
            rotation=0,
        )
        plt.title("Agent Win Rates")


def main():
    config = ExperimentConfig()
    agent_loader = AgentLoader(config)
    evaluator = AgentEvaluator(config)
    visualizer = ResultVisualizer()

    # Load agents
    checkpoint_files = sorted(os.listdir(config.checkpoint_dir))[::3]
    agents = [
        agent_loader.load_agent(f"{config.checkpoint_dir}/{cp}")
        for cp in checkpoint_files
    ]

    # Run tournament
    winrates = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
    for (i, agent1), (j, agent2) in combinations(enumerate(agents), 2):
        wins = evaluator.run_matchup(agent1, agent2)
        print(f"{agent1.id} vs {agent2.id}: {wins}")
        winrates[i][j] = wins[agent1.id] / config.num_games
        winrates[j][i] = wins[agent2.id] / config.num_games

    # Visualize results
    visualizer.create_heatmap(winrates, checkpoint_files)


if __name__ == "__main__":
    main()
