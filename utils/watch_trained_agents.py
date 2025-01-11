import torch
from generals.envs import PettingZooGenerals
from supervised.network import Network
from supervised.neuro_tensor import NeuroAgent
from generals import GridFactory
from generals.core.action import compute_valid_move_mask

n_envs = 1


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
    agent = NeuroAgent(model, id=path.split("/")[-1].split(".")[0], batch_size=n_envs)
    return agent


gf = GridFactory(
    min_grid_dims=(10, 10),
    max_grid_dims=(24, 24),
)

agents = {
    "sup114": load_agent("checkpoints/sup114/epoch=0-step=52000.ckpt"),
    "sup118": load_agent("checkpoints/sup118/epoch=0-step=24000.ckpt"),
}
agent_names = ["sup114", "sup118"]

# Create environment
n_games = 19
wins = {agent: 0 for agent in agent_names}
for g in range(10):
    done = False
    for agent in agents.values():
        agent.reset()
    env = PettingZooGenerals(agents=agent_names, grid_factory=gf, render_mode="human")
    observations, info = env.reset()
    while not done:
        actions = {}
        for agent in env.agents:  # go over agent ids
            # Ask each agent for his action
            obs = observations[agent]
            mask = compute_valid_move_mask(obs)[None, ...]
            obs = obs.as_tensor()[None, ...]
            action = agents[agent].act(obs, mask)[0]
            actions[agent] = action
        # All agents perform their actions
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = any(terminated.values()) or any(truncated.values())
        # detemine winner
        if done:
            print(info)
            for agent in agents:
                if info[agent]["is_winner"]:
                    wins[agent] += 1
                    print(wins)
                    print(f"Agent {agent} won game {g}")
        env.render()

print(wins)
