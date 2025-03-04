from supervised.agent import load_agent, load_fabric_checkpoint
from generals import PettingZooGenerals, GridFactory
from generals.core.rewards import RewardFn, compute_num_generals_owned
from generals.core.observation import Observation
from generals.core.action import Action
import numpy as np

n_envs = 1

class ShapedRewardFn(RewardFn):
    """A reward function that shapes the reward based on the number of generals owned."""

    def __init__(self, clip_value: float = 1.5, shaping_weight: float = 0.5):
        self.maximum_ratio = clip_value
        self.shaping_weight = shaping_weight

    def calculate_ratio_reward(self, my_army: int, opponent_army: int) -> float:
        ratio = my_army / opponent_army
        ratio = np.log(ratio) / np.log(self.maximum_ratio)
        return np.minimum(np.maximum(ratio, -1.0), 1.0)

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

        # If the game is done, we dont want to shape the reward
        if obs.owned_army_count == 0 or obs.opponent_army_count == 0:
            return original_reward

        prev_ratio_reward = self.calculate_ratio_reward(prior_obs.owned_army_count, prior_obs.opponent_army_count)
        current_ratio_reward = self.calculate_ratio_reward(obs.owned_army_count, obs.opponent_army_count)

        return float(original_reward + self.shaping_weight * (current_ratio_reward - prev_ratio_reward))


# agent1 = load_fabric_checkpoint("checkpoints/selfplay/against_snowballer.ckpt", mode="online")
# agent2 = load_fabric_checkpoint("checkpoints/selfplay/snowballer.ckpt", mode="online")
agent1 = load_fabric_checkpoint("checkpoints/selfplay/hehe.ckpt", mode="online")
agent2 = load_fabric_checkpoint("checkpoints/selfplay/step=48000.ckpt", mode="online")
# agent2 = load_agent("checkpoints/sup335/step=50000.ckpt", mode="online")

agent_names = ["hehe", "48k"]
agents = {agent_names[0]: agent1, agent_names[1]: agent2}

# Create environment
n_games = 50
gf = GridFactory(mode="generalsio")
wins = {agent: 0 for agent in agent_names}
for g in range(n_games):
    done = False
    for agent in agents.values():
        agent.reset()
    env = PettingZooGenerals(agents=agent_names, grid_factory=gf, render_mode="human", reward_fn=ShapedRewardFn())
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
            for agent in agents:
                if info[agent]["is_winner"]:
                    wins[agent] += 1
                    print(f"Agent {agent} won game {g} .. {wins}")
        env.render()

print(wins)
