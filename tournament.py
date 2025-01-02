import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generals import GridFactory
from supervised.network import Network
from supervised.neuro_agent import NeuroAgent
from utils.tournament import matchup


def load_agent(path):
    # map location based on the availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = torch.load(path, map_location=device)
    state_dict = network["state_dict"]
    model = Network(lr=1e-4, n_steps=9, compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)
    model.eval()
    agent = NeuroAgent(model, id=path.split("/")[-1].split(".")[0])
    return agent


def main():
    min_size = 15
    max_size = 23
    n_matches = 20

    grid_factory = GridFactory(
        min_grid_dims=(
            min_size,
            min_size,
        ),  # Grid height and width are randomly selected
        max_grid_dims=(max_size, max_size),
        mountain_density=0.08,  # Expected percentage of mountains
        city_density=0.05,  # Expected percentage of cities
        seed=38,  # Seed to generate the same map every time
    )
    agents = [path for path in os.listdir("checkpoints/sup109")]
    winrates = [[0 for _ in range(len(agents))] for _ in range(len(agents))]
    for agent1_path in agents:
        agent1 = load_agent(f"checkpoints/sup109/{agent1_path}")
        for agent2_path in agents:
            if agent1_path == agent2_path:
                continue
            agent2 = load_agent(f"checkpoints/sup109/{agent2_path}")
            wins = matchup(agent1, agent2, grid_factory, num_games=n_matches)
            print(f"{agent1.id} vs {agent2.id}: {wins}")
            winrates[agents.index(agent1)][agents.index(agent2)] = wins[agent1.id]

    mask = np.eye(
        len(agents), dtype=bool
    )  # True for diagonal elements, False otherwise

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)  # Red for small, Green for large
    sns.heatmap(
        winrates,
        annot=True,  # Show values on the heatmap
        fmt="g",  # General number format for annotation
        cmap=cmap,  # Custom colormap
        mask=mask,  # Explicit mask for diagonal
        cbar_kws={"label": "Wins"},  # Label for the colorbar
    )
    plt.xticks(
        ticks=np.arange(len(agents)) + 0.5,
        labels=[agent.id for agent in agents],
        rotation=45,
    )
    plt.yticks(
        ticks=np.arange(len(agents)) + 0.5,
        labels=[agent.id for agent in agents],
        rotation=0,
    )
    plt.title("Winrates of agents")
    plt.savefig("winrates.png")
    plt.show()


if __name__ == "__main__":
    main()
