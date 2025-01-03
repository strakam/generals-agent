import torch
import gymnasium as gym
import numpy as np
from generals.envs import MultiAgentGymnasiumGenerals
from generals import GridFactory
from supervised.network import Network
from supervised.neuro_tensor import NeuroAgent


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
    agent = NeuroAgent(model, id=path.split("/")[-1].split(".")[0], batch_size=12)
    return agent


agents = {
    "agent1": load_agent("checkpoints/sup114/epoch=0-step=4000.ckpt"),
    "agent2": load_agent("checkpoints/sup114/epoch=0-step=52000.ckpt"),
}
agent_names = ["agent1", "agent2"]

gf = GridFactory(
    min_grid_dims=(10, 10),
    max_grid_dims=(24, 24),
    mountain_density=0.08,
    city_density=0.05,
    seed=38,
)

n_envs = 12
envs = gym.vector.AsyncVectorEnv(
    [
        lambda: MultiAgentGymnasiumGenerals(
            agents=agent_names, grid_factory=gf, truncation=500
        )
        for _ in range(n_envs)
    ],
)


observations, infos = envs.reset()
terminated = [False] * len(observations)
truncated = [False] * len(observations)

while True:
    agent_1_obs = observations[:, 0, ...]
    agent_2_obs = observations[:, 1, ...]
    print(agent_1_obs.shape)
    masks = [
        np.stack([info[-1] for info in infos[agent_name]]) for agent_name in agent_names
    ]
    print(type(agent_1_obs))
    agent_1_actions = agents["agent1"].act(agent_1_obs, masks[0])
    print(agent_1_actions)
    # agent_2_actions = [agent.act(obs) for agent, obs in zip(agents["agent2"], observations)]
    # actions = np.stack([agent_1_actions, agent_2_actions], axis=1)
    # observations, rewards, terminated, truncated, infos = envs.step(actions)
    break
    if any(terminated) or any(truncated):
        print("DONE!")
        break
