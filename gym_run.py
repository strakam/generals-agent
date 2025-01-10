import neptune
import numpy as np
import gymnasium as gym
import torch
from generals.envs import GymnasiumGenerals
from supervised.network import Network
from supervised.neuro_tensor import NeuroAgent
from generals import GridFactory

n_envs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

key_file = open("neptune_token.txt", "r")
key = key_file.read()
run = neptune.init_run(
    project="strakam/supervised-agent",
    api_token=key,
)


def load_agent(path, channel_sequence=[256, 320, 384, 384]) -> NeuroAgent:
    # Map location based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = torch.load(path, map_location=device)
    state_dict = network["state_dict"]
    model = Network(channel_sequence=channel_sequence, compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)
    model.eval()
    agent = NeuroAgent(
        model, id=path.split("/")[-1].split(".")[0], batch_size=n_envs, device=device
    )
    return agent


gf = GridFactory(
    min_grid_dims=(15, 15),
    max_grid_dims=(24, 24),
)

agents = {
    "agent1": load_agent(
        "checkpoints/epoch=0-step=40000.ckpt",
        channel_sequence=[320, 384, 448, 448],
    ),
    "agent2": load_agent("checkpoints/epoch=0-step=52000.ckpt"),
}
agent_names = ["agent1", "agent2"]

# Create environment
envs = gym.vector.AsyncVectorEnv(
    [
        lambda: GymnasiumGenerals(
            agent_ids=agent_names, grid_factory=gf, truncation=1500, pad_to=24
        )
        for _ in range(n_envs)
    ],
)
observations, infos = envs.reset()

done = False
t = 0
ended = 0
wins = {agent: 0 for agent in agent_names}
while ended < 100:
    # swap first two axes of observations
    agent_1_obs = observations[:, 0, ...]
    agent_2_obs = observations[:, 1, ...]
    masks = [np.stack([info[-1] for info in infos[agent]]) for agent in agent_names]
    agent_1_actions = agents["agent1"].act(agent_1_obs, masks[0])
    agent_2_actions = agents["agent2"].act(agent_2_obs, masks[1])
    actions = np.stack([agent_1_actions, agent_2_actions], axis=1)
    observations, rewards, terminated, truncated, infos = envs.step(actions)
    t += 1
    done = any(terminated) or any(truncated)
    if done and any(terminated):
        for agent in agent_names:
            outputs = infos[agent]
            for game in outputs:
                if game[3] == 1:
                    wins[agent] += 1
                    ended += 1
                    other_agent = [a for a in agent_names if a != agent][0]
                    run["winrate/{}".format(agent)].log(wins[agent] / ended)
                    run["winrate/{}".format(other_agent)].log(wins[other_agent] / ended)
        print(wins)
    print(f"Time {t}, ended {ended}")
print(wins)
