import torch
from generals.agents import ExpanderAgent
from generals.envs import PettingZooGenerals

# Initialize agents
expander = ExpanderAgent()

# Store agents in a dictionary - they are called by id, which will come handy
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
    expander.id
}

agents = {
    random.id: random,
    expander.id: expander,
}

# Create environment
env = PettingZooGenerals(agents=agents, render_mode="human")
observations, info = env.reset()

done = False
while not done:
    actions = {}
    for agent in env.agents: # go over agent ids
        # Ask each agent for his action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions) # perform all actions
    done = any(terminated.values()) or any(truncated.values())
    env.render()
