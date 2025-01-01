import os
import torch
from generals.envs import PettingZooGenerals
from generals import GridFactory
from neuro_agent import NeuroAgent
from network import Network
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time


def matchup(agent1, agent2, gf, num_games=10):
    agents = {
        agent1.id: agent1,
        agent2.id: agent2,
    }
    env = PettingZooGenerals(agents=agents, grid_factory=gf, render_mode=None)
    win_count = {agent1.id: 0, agent2.id: 0}
    print(f"{agent1.id} vs {agent2.id}")
    for _ in tqdm.tqdm(range(num_games)):
        observations, _ = env.reset()
        done = False
        t=0
        time_spend_thinking = 0
        game_start = time.time()
        while not done:
            actions = {}
            start = time.time()
            for agent in env.agents:
                actions[agent] = agents[agent].act(observations[agent])
            think = time.time() - start
            print(think)
            time_spend_thinking += think
            observations, _, terminated, truncated, infos = env.step(actions)
            done = any(terminated.values()) or any(truncated.values())
            t+=1
        for agent in agents:
            if infos[agent]["is_winner"]:
                win_count[agent] += 1
        print(f"Game length {t}.")
        print(f"Time spend thinking: {time_spend_thinking}")
        print(f"Game time: {time.time() - game_start}")
    return win_count


agent_checkpoints = [
    os.path.join("checkpoints2", f) for f in os.listdir("checkpoints2")
]
agent_list = []
for checkpoint in agent_checkpoints:
    checkpoint_id = checkpoint.split("/")[-1].split(".")[0]
    network = torch.load(checkpoint, map_location="cpu")
    state_dict = network["state_dict"]
    model = Network(lr=1e-4, n_steps=9, input_dims=(29, 24, 24), compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)
    # generate random color
    color = tuple(np.random.randint(0, 255, 3))
    agent_list.append(NeuroAgent(model, id=checkpoint_id, color=color))


min_size = 15
max_size = 23
n_matches = 20

grid_factory = GridFactory(
    min_grid_dims=(min_size, min_size),  # Grid height and width are randomly selected
    max_grid_dims=(max_size, max_size),
    mountain_density=0.08,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    seed=38,  # Seed to generate the same map every time
)

winrates = [[0 for _ in range(len(agent_list))] for _ in range(len(agent_list))]
print("Agents loaded!")
for agent1 in agent_list:
    for agent2 in agent_list:
        if agent1 == agent2:
            continue
        wins = matchup(agent1, agent2, grid_factory, num_games=n_matches)
        print(f"{agent1.id} vs {agent2.id}: {wins}")
        winrates[agent_list.index(agent1)][agent_list.index(agent2)] = wins[agent1.id]

mask = np.eye(
    len(agent_list), dtype=bool
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
    ticks=np.arange(len(agent_list)) + 0.5,
    labels=[agent.id for agent in agent_list],
    rotation=45,
)
plt.yticks(
    ticks=np.arange(len(agent_list)) + 0.5,
    labels=[agent.id for agent in agent_list],
    rotation=0,
)
plt.title("Winrates of agents")
plt.show()
