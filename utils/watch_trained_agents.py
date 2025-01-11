import torch
from generals.envs import PettingZooGenerals
from generals import GridFactory
from neuro_agent import NeuroAgent
from neuro2 import Neuro2Agent
from network import Network
from network2 import Network2

def load_network(path, channels):
    network = torch.load(path, map_location="cpu")
    state_dict = network["state_dict"]
    if channels == 29:
        model = Network2(
            lr=1e-4, n_steps=9, input_dims=(channels, 24, 24), compile=True
        )
    else:
        model = Network(
            lr=1e-4, n_steps=9, input_dims=(channels, 24, 24), compile=True
        )
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict)
    return model

# Initialize agents
neuro = Neuro2Agent(load_network("checkpoints2/epoch=0-step=78000.ckpt", 29), id="Yesterday", color=(0, 30, 220))
neuro2 = NeuroAgent(load_network("checkpoints/epoch=0-step=78000.ckpt", 31), id="Today")

net = torch.load("checkpoints/epoch=0-step=78000.ckpt", map_location="cpu")
print(net["optimizer_states"])

exit()
grid_factory = GridFactory(
    min_grid_dims=(10, 10),  # Grid height and width are randomly selected
    max_grid_dims=(15, 15),
    mountain_density=0.13,  # Expected percentage of mountains
    city_density=0.09,  # Expected percentage of cities
    general_positions=[(1, 1), (8, 8)],  # Positions of the generals
    seed=38,  # Seed to generate the same map every time
)

# Store agents in a dictionary
agents = {
    neuro.id: neuro,
    neuro2.id: neuro2,
}

# Create environment
wins = 0
for i in range(20):
    env = PettingZooGenerals(agents=agents, grid_factory=grid_factory, render_mode="human")
    observations, info = env.reset()
    neuro.reset()
    done = False
    while not done:
        actions = {}
        for agent in env.agents:
            # Ask agent for action
            actions[agent] = agents[agent].act(observations[agent])
        # All agents perform their actions
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = any(terminated.values()) or any(truncated.values())
        env.render()
        if done:
            if info[neuro.id]["is_winner"]:
                wins += 1
                print("Win")
print(wins)
