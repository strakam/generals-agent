from supervised.agent import load_network, load_fabric_checkpoint, EnsembleAgent
from generals import PettingZooGenerals, GridFactory
from generals.core.rewards import RewardFn, compute_num_generals_owned
from generals.core.observation import Observation
from generals.core.action import Action
import numpy as np

n_envs = 1


# agent1 = load_fabric_checkpoint("checkpoints/selfplay/against_snowballer.ckpt", mode="online")
# agent2 = load_fabric_checkpoint("checkpoints/selfplay/snowballer.ckpt", mode="online")
agent1 = load_fabric_checkpoint("checkpoints/experiments/1061-21.ckpt", mode="online")
# agent2 = load_fabric_checkpoint("checkpoints/experiments/1153-7.ckpt", mode="online")
# agent2 = load_agent("checkpoints/sup335/step=50000.ckpt", mode="online")

networks = [load_network("checkpoints/experiments/zero3.ckpt"), load_network("checkpoints/experiments/special.ckpt")]
agent2 = EnsembleAgent(networks=networks)

agent_names = ["21", "7"]
agents = {agent_names[0]: agent1, agent_names[1]: agent2}

# Create environment
n_games = 250
gf = GridFactory(mode="generalsio")
wins = {agent: 0 for agent in agent_names}
for g in range(n_games):
    done = False
    for agent in agents.values():
        agent.reset()
    env = PettingZooGenerals(agents=agent_names, grid_factory=gf, render_mode="human")
    observations, info = env.reset(options={"replay_file": "hehe"})
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
            for agent in agents:
                if info[agent]["is_winner"]:
                    wins[agent] += 1
                    print(f"Agent {agent} won game {g} .. {wins}")
        env.render()

print(wins)
