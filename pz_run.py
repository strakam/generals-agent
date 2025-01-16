import torch
from checkpoints.neuro_tensor import OnlineAgent as OnlineAgent1
from checkpoints.network import Network as Network1
from supervised.network import Network
from supervised.agent import OnlineAgent
from generals import PettingZooGenerals, GridFactory

n_envs = 1

# Map location based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = torch.load("checkpoints/sup114/epoch=0-step=40000.ckpt", map_location=device)
state_dict = network["state_dict"]
model = Network1(compile=True)
model_keys = model.state_dict().keys()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
model.load_state_dict(filtered_state_dict)
model.eval()
agent1 = OnlineAgent1(model, id=0, device=device)

net1 = torch.load("checkpoints/sup169/epoch=0-step=32000.ckpt", map_location=device)
model1 = Network(compile=True)
model1.load_state_dict(net1["state_dict"])
model1.eval()
agent2 = OnlineAgent(model1, id=1, device=device)

agent_names = ["sup114", "sup169"]
agents = {"sup114": agent1, "sup169": agent2}

# Create environment
n_games = 19
gf = GridFactory(
    min_grid_dims=(15, 15),
    max_grid_dims=(24, 24),
)
wins = {agent: 0 for agent in agent_names}
for g in range(10):
    done = False
    for agent in agents.values():
        agent.reset()
    env = PettingZooGenerals(agents=agent_names, grid_factory=gf, render_mode="human")
    observations, info = env.reset()
    agent1.reset()
    agent2.reset()
    while not done:
        actions = {}
        for agent in env.agents:  # go over agent ids
            # Ask each agent for his action
            obs = observations[agent]
            action = agents[agent].act(obs)
            actions[agent] = action
        # All agents perform their actions
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
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
