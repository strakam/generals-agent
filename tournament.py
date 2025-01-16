import json
import gymnasium as gym
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict
from torch import nn
import neptune

from generals import GridFactory
from generals.envs import GymnasiumGenerals
from supervised.network import Network
from supervised.agent import NeuroAgent


@dataclass
class ExperimentConfig:
    """Configuration parameters for the experiment."""

    n_envs: int = 256
    num_games: int = 250
    checkpoint_dir: str = "checkpoints/sup196"
    min_grid_size: int = 15
    max_grid_size: int = 23
    mountain_density: float = 0.15
    city_density: float = 0.03
    truncation_steps: int = 1500
    channel_sequence: List[int] = (256, 320, 384, 384)
    neptune_project: str = "strakam/supervised-agent"
    output_directory: str = "/storage/praha1/home/strakam3"


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
            agents=agent_names,
            grid_factory=grid_factory,
            truncation=self.config.truncation_steps,
            pad_observations_to=24,
        )


class AgentLoader:
    """Handles loading and initialization of agents."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_agent(self, checkpoint_path: str) -> NeuroAgent:
        print(f"Loading agent from {checkpoint_path}")
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


class NeptuneLogger:
    """Handles logging experiment metrics to Neptune."""

    def __init__(self, config: ExperimentConfig):
        self.run = neptune.init_run(
            project=config.neptune_project,
            name="generals-tournament",
            tags=["tournament", "evaluation"],
        )
        self.config = config
        self._log_config()

    def _log_config(self):
        """Log experiment configuration parameters."""
        self.run["parameters"] = {
            "n_envs": self.config.n_envs,
            "num_games": self.config.num_games,
            "grid_size": f"{self.config.min_grid_size}-{self.config.max_grid_size}",
            "mountain_density": self.config.mountain_density,
            "city_density": self.config.city_density,
            "truncation_steps": self.config.truncation_steps,
        }

    def log_game_result(self, agent1_id: str, agent2_id: str, winner_id: str):
        """Log individual game results."""
        matchup = f"{agent1_id}_vs_{agent2_id}"
        self.run[f"games/{matchup}/results"].append({"winner": winner_id})

    def log_matchup_results(self, agent1_id: str, agent2_id: str, wins: Dict[str, int]):
        """Log final results of a matchup between two agents."""
        matchup = f"{agent1_id}_vs_{agent2_id}"
        total_games = self.config.num_games
        self.run[f"matchups/{matchup}"] = {
            f"{agent1_id}_wins": wins[agent1_id],
            f"{agent2_id}_wins": wins[agent2_id],
            f"{agent1_id}_winrate": wins[agent1_id] / total_games,
            f"{agent2_id}_winrate": wins[agent2_id] / total_games,
        }

    def log_heatmap(self, heatmap_path: str):
        """Log the winrate heatmap visualization."""
        self.run["visualizations/winrate_heatmap"].upload(heatmap_path)

    def close(self):
        """Close the Neptune run."""
        self.run.stop()


class AgentEvaluator:
    """Handles agent matchups and evaluations."""

    def __init__(self, config: ExperimentConfig, logger: NeptuneLogger):
        self.config = config
        self.env_factory = EnvironmentFactory(config)
        self.logger = logger

    def run_matchup(self, agent1: NeuroAgent, agent2: NeuroAgent) -> Dict[str, int]:
        names = [agent1.id, agent2.id]
        envs = gym.vector.AsyncVectorEnv(
            [lambda: self._env_creator(names) for _ in range(self.config.n_envs)]
        )

        observations, infos = envs.reset()
        wins = {agent1.id: 0, agent2.id: 0}
        ended_games = 0

        while ended_games < self.config.num_games:
            # for each agent, get the action
            actions = []
            for i, agent in enumerate([agent1, agent2]):
                masks = np.stack([info[-1] for info in infos[agent.id]])
                actions.append(agent.act(observations[:, i, ...], masks))

            actions = np.stack(actions, axis=1)

            # game step
            observations, _, terminated, _, infos = envs.step(actions)
            if any(terminated):
                ended_games += self._process_game_results(
                    infos, wins, agent1.id, agent2.id
                )

        self.logger.log_matchup_results(agent1.id, agent2.id, wins)
        return wins

    def _env_creator(self, names: List[str]) -> GymnasiumGenerals:
        return self.env_factory.create_environment(names)

    def _process_game_results(
        self, infos: Dict, wins: Dict[str, int], agent1_id: str, agent2_id: str
    ) -> int:
        ended = 0
        for agent_id in wins.keys():
            for game in infos[agent_id]:
                if game[3] == 1:  # Victory condition
                    print(f"{agent_id} won!")
                    wins[agent_id] += 1
                    ended += 1
                    self.logger.log_game_result(agent1_id, agent2_id, agent_id)
        return ended


class ResultVisualizer:
    """Handles visualization of tournament results."""

    def create_heatmap(
        self,
        winrates: List[List[float]],
        agent_names: List[str],
        output_path: str = "tournament_heatmap.png",
        # output_path: str = "tournament_heatmap.png",
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
        return output_path

    def _configure_plot(self, agent_names: List[str]):
        labels = [name.split(".")[0].split("=")[-1] for name in agent_names]
        labels = [str(int(label) // 1000) for label in labels]
        plt.xticks(
            ticks=np.arange(len(agent_names)) + 0.5,
            labels=labels,
            rotation=45,
        )
        plt.yticks(
            ticks=np.arange(len(agent_names)) + 0.5,
            labels=labels,
            rotation=0,
        )
        plt.title("Agent Win Rates")


def main():
    config = ExperimentConfig()
    neptune_logger = NeptuneLogger(config)
    agent_loader = AgentLoader(config)
    evaluator = AgentEvaluator(config, neptune_logger)
    visualizer = ResultVisualizer()

    try:
        # Load agents
        checkpoint_files = sorted(os.listdir(config.checkpoint_dir))
        print(checkpoint_files)
        agents = [
            agent_loader.load_agent(f"{config.checkpoint_dir}/{cp}")
            for cp in checkpoint_files
        ]

        # Run tournament
        winrates = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
        for (i, agent1), (j, agent2) in combinations(enumerate(agents), 2):
            wins = evaluator.run_matchup(agent1, agent2)
            winrates[i][j] = wins[agent1.id] / config.num_games
            winrates[j][i] = wins[agent2.id] / config.num_games

        print(winrates)
        # Visualize results
        path = f"{config.output_directory}/winrates.json"
        json.dump(winrates, open(path, "w"))
        heatmap_path = visualizer.create_heatmap(
            winrates,
            checkpoint_files,
            f"{config.output_directory}/tournament_heatmap.png",
        )
        neptune_logger.log_heatmap(heatmap_path)

    finally:
        neptune_logger.close()


if __name__ == "__main__":
    main()
