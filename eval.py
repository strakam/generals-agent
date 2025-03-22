import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
import gymnasium as gym
from lightning.fabric import Fabric

from generals import GridFactory, GymnasiumGenerals
from selfplay.network import load_fabric_checkpoint
from selfplay.logger import NeptuneLogger
torch.set_float32_matmul_precision("medium")

@dataclass
class SelfPlayConfig:
    # Training parameters
    training_iterations: int = 1000
    n_envs: int = 256
    n_steps: int = 3
    n_epochs: int = 4
    batch_size: int = 6
    truncation: int = 2000
    grid_size: int = 23
    channel_sequence: List[int] = field(default_factory=lambda: [256, 256, 288, 288])
    repeats: List[int] = field(default_factory=lambda: [2, 2, 2, 1])
    checkpoint_path: str = ""

    # PPO parameters
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    learning_rate: float = 2.5e-5  # Standard PPO learning rate
    max_grad_norm: float = 0.25  # Gradient clipping
    clip_coef: float = 0.2  # PPO clipping coefficient
    ent_coef: float = 0.004  # Increased from 0.00 to encourage exploration
    vf_coef: float = 0.3  # Value function coefficient
    target_kl: float = 0.02  # Target KL divergence
    norm_adv: bool = True  # Whether to normalize advantages

    # Lightning fabric parameters
    strategy: str = "auto"
    precision: str = "bf16-mixed"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 42


def create_environment(cfg: SelfPlayConfig) -> gym.vector.AsyncVectorEnv:
    # Create environments with different min_generals_distance values
    envs = []
    for i in range(cfg.n_envs):
        # Use min_generals_distance=15 for all environments
        envs.append(
            lambda: GymnasiumGenerals(
                agents=["1", "2"],
                grid_factory=GridFactory(mode="generalsio"),
                truncation=cfg.truncation,
                pad_observations_to=24,
            )
        )

    return gym.vector.AsyncVectorEnv(
        envs,
        shared_memory=True,
    )

class SelfPlayTrainer:
    """Encapsulates setup and training loop for self-play."""

    def __init__(self, cfg: SelfPlayConfig, agent_names: List[str]):
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

        self.agents = {}
        agent_paths = [f"checkpoints/experiments/{name}.ckpt" for name in agent_names]
        for name in agent_names:
            agent = load_fabric_checkpoint(agent_paths[agent_names.index(name)], batch_size=self.cfg.n_envs, eval_mode=True)
            agent = self.fabric.setup(agent)
            self.agents[name] = agent

        # Create environment.
        self.envs = create_environment(cfg)

        # Setup expected tensor shapes for rollouts.
        self.n_agents = 2


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
            obs = torch.from_numpy(obs)
            obs = obs.to(self.fabric.device, non_blocking=True)

            # Create augmented observations for both players
            augmented_obs = torch.stack(
                [
                    self.ag1.augment_observation(obs[:, 0, ...]),
                    self.ag2.augment_observation(obs[:, 1, ...]),
                ],
                dim=1,
            )

            # Process masks.
            _masks = torch.zeros((self.cfg.n_envs, self.n_agents, 24, 24, 4), device=self.fabric.device)
            for i, agent_name in enumerate(["1", "2"]):
                masks = torch.from_numpy(infos[agent_name]["masks"])
                masks = masks.to(self.fabric.device, non_blocking=True)
                _masks[:, i] = masks
            # Process rewards.
            rewards = torch.stack(
                [
                    torch.from_numpy(infos[agent_name]["reward"]).to(self.fabric.device, non_blocking=True)
                    for agent_name in ["1", "2"]
                ],
                dim=1,
            ).reshape(self.cfg.n_envs, self.n_agents)

        return augmented_obs, _masks, rewards

    def run(self, agent1, agent2):
        """Runs the main training loop."""
        # Pre-allocate action arrays to avoid repeated numpy allocations
        self.ag1 = self.agents[agent1]
        self.ag2 = self.agents[agent2]
        self.ag1.reset()
        self.ag2.reset()
        _actions = np.empty((self.cfg.n_envs, 2, 5), dtype=np.int32)

        next_obs, infos = self.envs.reset()
        prev_obs = next_obs
        next_obs, mask, _ = self.process_observations(next_obs, infos)

        wins, draws, losses, avg_game_length = 0, 0, 0, 0
        step = 0
        n_games = 250

        # Continue collecting data until we have enough completed games
        while (wins + losses) < n_games:
            with torch.no_grad(), self.fabric.device:
                # Get actions for player 1 (learning player)
                player1_obs = next_obs[:, 0]
                player1_mask = mask[:, 0]
                player1_actions, _ = self.ag1.predict(player1_obs, player1_mask)
                _actions[:, 0] = player1_actions.cpu().numpy()

                # Get actions for player 2 (fixed network) without storing
                player2_obs = next_obs[:, 1]
                player2_mask = mask[:, 1]
                player2_actions, _ = self.ag2.predict(player2_obs, player2_mask)
                _actions[:, 1] = player2_actions.cpu().numpy()

            next_obs, _, terminations, truncations, infos = self.envs.step(_actions)
            dones = np.logical_or(terminations, truncations)

            # Track game outcomes when episodes finish
            if any(dones):
                game_times = prev_obs[:, 0, 13, 0, 0]  # Get game times from observations
                p1_wins = infos["1"]["winner"]
                p2_wins = infos["2"]["winner"]
                wins += np.sum(dones & p1_wins)
                losses += np.sum(dones & p2_wins)
                draws += np.sum(dones & ~p1_wins & ~p2_wins)
                # Calculate total game time for completed games
                avg_game_length += np.sum(game_times[dones])

            next_obs, mask, _ = self.process_observations(next_obs, infos)
            step += 1

        # Calculate win/draw/loss percentages
        total_games = wins + draws + losses
        win_rate = wins / total_games
        draw_rate = draws / total_games
        loss_rate = losses / total_games

        self.fabric.print(
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
        return win_rate, draw_rate, loss_rate


def main():
    agent_names = ["anti", "castler2", "zero0", "zero1", "zero2", "zero3", "cp_79"]
    cfg = SelfPlayConfig()
    trainer = SelfPlayTrainer(cfg, agent_names)


    # Create winrate matrix
    n_agents = len(agent_names)
    winrate_matrix = np.zeros((n_agents, n_agents))

    from itertools import combinations
    for i, (agent1, agent2) in enumerate(combinations(agent_names, 2)):
        win_rate, draw_rate, loss_rate = trainer.run(agent1, agent2)
        print(f"{agent1} vs {agent2} - P1 win rate: {win_rate:.2%}, Draw rate: {draw_rate:.2%}, P2 win rate: {loss_rate:.2%}")
        
        # Fill the matrix (both symmetric positions)
        i1, i2 = agent_names.index(agent1), agent_names.index(agent2)
        winrate_matrix[i1, i2] = win_rate
        winrate_matrix[i2, i1] = loss_rate  # Note: loss_rate for agent1 is win_rate for agent2

    # Fill diagonal with 0.5 (assuming draw when same agent plays against itself)
    np.fill_diagonal(winrate_matrix, 0.5)

    # Create heatmap
    plt.figure(figsize=(8, 6))
    mask = np.eye(n_agents, dtype=bool)  # Create mask for diagonal entries
    
    # First create the main heatmap without diagonal
    sns.heatmap(
        winrate_matrix,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        xticklabels=agent_names,
        yticklabels=agent_names,
        cbar_kws={'label': 'Win Rate'},
        mask=mask  # Mask out diagonal entries
    )
    
    # Add black diagonal entries
    sns.heatmap(
        np.eye(n_agents),
        cmap=['black'],
        cbar=False,
        xticklabels=agent_names,
        yticklabels=agent_names,
        mask=~mask  # Inverse mask to only show diagonal
    )
    
    plt.title('Win Rate Matrix')
    plt.tight_layout()
    plt.savefig('winrate_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()
