from generals.agents import ExpanderAgent, RandomAgent
from generals.envs import PettingZooGenerals
from generals import GridFactory
import seaborn as sns
import matplotlib.pyplot as plt


def matchup(agent1, agent2, gf, num_games=10):
    agents = {
        agent1.id: agent1,
        agent2.id: agent2,
    }
    env = PettingZooGenerals(agents=agents, grid_factory=gf, render_mode="none")
    win_count = {agent1.id: 0, agent2.id: 0}
    for _ in range(num_games):
        observations, _ = env.reset()
        done = False
        while not done:
            actions = {}
            for agent in env.agents:
                actions[agent] = agents[agent].act(observations[agent])
            observations, _, terminated, truncated, infos = env.step(actions)
            done = any(terminated.values()) or any(truncated.values())
        for agent in agents:
            if infos[agent]["is_winner"]:
                win_count[agent] += 1
    return win_count


# Initialize agents - must have unique ID!
a1 = ExpanderAgent(id="Expander")
a2 = RandomAgent(id="Random")
a3 = RandomAgent(id="Muhehe")
a4 = ExpanderAgent(id="Expander2")
a5 = ExpanderAgent(id="Pokemon")

min_size = 5
max_size = 6

grid_factory = GridFactory(
    min_grid_dims=(min_size, min_size),  # Grid height and width are randomly selected
    max_grid_dims=(max_size, max_size),
    mountain_density=0.08,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    seed=38,  # Seed to generate the same map every time
)

agent_list = [a1, a2, a3, a4, a5]
winrates = [[0 for _ in range(len(agent_list))] for _ in range(len(agent_list))]

for agent1 in agent_list:
    for agent2 in agent_list:
        if agent1 == agent2:
            continue
        wins = matchup(agent1, agent2, grid_factory, num_games=10)
        print(f"{agent1.id} vs {agent2.id}: {wins}")
        winrates[agent_list.index(agent1)][agent_list.index(agent2)] = wins[agent1.id]

print(winrates)
# plot the winrates, annotate with agent ids
plt.figure(figsize=(10, 7))
# make custom fontsize, but not only numbers but also agent ids
fontsize = 15
sns.heatmap(
    winrates,
    annot=True,
    xticklabels=[a.id for a in agent_list],
    yticklabels=[a.id for a in agent_list],
    annot_kws={"size": fontsize},
)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()


