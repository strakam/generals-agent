import torch
from generals.agents import ExpanderAgent
from generals.envs import PettingZooGenerals
from generals import GridFactory
from neuro_agent import NeuroAgent
from network import Network

network = torch.load("checkpoints2/epoch=0-step=60000.ckpt", map_location="cpu")
state_dict = network["state_dict"]
model = Network(
    lr=1e-4, n_steps=9, input_dims=(29, 24, 24), compile=True
)
model_keys = model.state_dict().keys()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
model.load_state_dict(filtered_state_dict)

# Initialize agents
neuro = NeuroAgent(model)
expander = ExpanderAgent()

grid_factory = GridFactory(
    min_grid_dims=(15, 15),  # Grid height and width are randomly selected
    max_grid_dims=(20, 20),
    mountain_density=0.08,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    general_positions=[(1, 2), (12, 13)],  # Positions of the generals
    seed=38,  # Seed to generate the same map every time
)

# Store agents in a dictionary
agents = {
    neuro.id: neuro,
    expander.id: expander,
}

# Create environment
env = PettingZooGenerals(agents=agents, grid_factory=grid_factory, render_mode="human")
observations, info = env.reset()

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
