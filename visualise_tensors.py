import torch
from utils.tensor_vis import visualize_tensor
from generals import Replay
from generals.envs import PettingZooGenerals
from generals.agents import ExpanderAgent
from generals import GridFactory
from supervised.neuro_agent import NeuroAgent

# Initialize agents
neuro1 = NeuroAgent(id="Agent1")
neuro2 = NeuroAgent(id="Agent2", color=(0, 0, 255))
expander = ExpanderAgent()

# Store agents in a dictionary
agents = {
    neuro1.id: neuro1,
    neuro2.id: neuro2,
}

gf = GridFactory(
    min_grid_dims=(4, 4),
    max_grid_dims=(8, 8),
)
# Create environment
env = PettingZooGenerals(agents=agents, render_mode=None, grid_factory=gf)
replay = "verify"
observations, info = env.reset(options={"replay_file": replay})

done = False
channel = 26
images = []
agent_to_visualize = "Agent1"
while not done:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        _ = agents[agent].act(observations[agent])
        actions[agent] = expander.act(observations[agent])
        if agent == agent_to_visualize:
            images.append(agents[agent].last_observation[channel])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values()) or any(truncated.values())

# convert images to tensor
images = torch.stack([torch.tensor(image) for image in images])
visualize_tensor(images)
