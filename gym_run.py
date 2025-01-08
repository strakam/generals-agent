import numpy as np
import gymnasium as gym
import torch
from generals.envs import MultiAgentGymnasiumGenerals
from supervised.network import Network
from supervised.neuro_tensor import NeuroAgent
from generals import GridFactory

n_envs = 2


# Store agents in a dictionary - they are called by id, which will come handy
def load_agent(path, channel_sequence=None):
    # Map location based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = torch.load(path, map_location=device)
    state_dict = network["state_dict"]
    if channel_sequence is not None:
        model = Network(lr=1e-4, n_steps=9, channel_sequence=channel_sequence, compile=True)
    else:
        model = Network(lr=1e-4, n_steps=9, compile=True)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)
    model.eval()
    agent = NeuroAgent(model, id=path.split("/")[-1].split(".")[0], batch_size=n_envs)
    return agent


gf = GridFactory(
    min_grid_dims=(10, 10),
    max_grid_dims=(24, 24),
)

agents = {
    "agent1": load_agent(
        "checkpoints/sup119/epoch=0-step=40000.ckpt",
        channel_sequence=[320, 384, 448, 448],
    ),
    "agent2": load_agent("checkpoints/sup114/epoch=0-step=52000.ckpt"),
}
agent_names = ["agent1", "agent2"]

# Create environment
envs = gym.vector.AsyncVectorEnv(
    [
        lambda: MultiAgentGymnasiumGenerals(
            agents=agent_names, grid_factory=gf, truncation=1500
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
        print(wins)
    print(f"Time {t}, ended {ended}")
print(wins)
