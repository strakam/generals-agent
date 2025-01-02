import os
from generals.envs import PettingZooGenerals
import torch
from generals import GridFactory
from supervised.network import Network
from supervised.neuro_agent import NeuroAgent
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor


def load_agent(path):
    # Map location based on availability
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


def one_game(agent1, agent2, gf):
    agents = {
        agent1.id: agent1,
        agent2.id: agent2,
    }
    env = PettingZooGenerals(agents=agents, grid_factory=gf)
    obs, _ = env.reset()
    while True:
        actions = {
            agent1.id: agent1.act(obs[agent1.id]),
            agent2.id: agent2.act(obs[agent2.id]),
        }
        obs, _, _, _, info = env.step(actions)
        if info[agent1.id]["is_done"]:
            if info[agent1.id]["is_winner"]:
                return agent1.id
            return agent2.id


def matchup(agent1, agent2, gf, num_games=10):
    win_count = {agent1.id: 0, agent2.id: 0}
    for _ in range(num_games):
        winner = one_game(agent1, agent2, gf)
        win_count[winner] += 1
    return win_count


def process_matchup(agent_paths, grid_factory, num_games, agent1_path, agent2_path):
    agent1 = load_agent(agent1_path)
    agent2 = load_agent(agent2_path)
    wins = matchup(agent1, agent2, grid_factory, num_games)
    return agent1.id, agent2.id, wins


def tournament():
    min_size = 10
    max_size = 15
    n_matches = 8

    grid_factory = GridFactory(
        min_grid_dims=(min_size, min_size),
        max_grid_dims=(max_size, max_size),
        mountain_density=0.08,
        city_density=0.05,
        seed=38,
    )

    agent_paths = [f"checkpoints/sup109/{path}" for path in os.listdir("checkpoints/sup109")]
    agent_paths.sort()
    agent_paths = agent_paths[1::2]
    agent_ids = [path.split("/")[-1].split(".")[0] for path in agent_paths]

    winrates = [[0 for _ in range(len(agent_ids))] for _ in range(len(agent_ids))]
    agent_pairs = list(combinations(agent_paths, 2))

    with ProcessPoolExecutor(num_workers=11) as executor:
        futures = [
            executor.submit(process_matchup, agent_paths, grid_factory, n_matches, a1, a2)
            for a1, a2 in agent_pairs
        ]

        for future in futures:
            agent1_id, agent2_id, wins = future.result()
            print(f"{agent1_id} vs {agent2_id}: {wins}")
            i1, i2 = agent_ids.index(agent1_id), agent_ids.index(agent2_id)
            winrates[i1][i2] = wins[agent1_id] / n_matches
            winrates[i2][i1] = wins[agent2_id] / n_matches

    print("Winrates:")
    for row in winrates:
        print(row)

if __name__ == "__main__":
    tournament()

