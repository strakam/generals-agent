import gymnasium as gym
import numpy as np
import time
import tqdm

from generals.agents import ExpanderAgent

# Initialize agents
agent = ExpanderAgent(id="kek")
npc = ExpanderAgent()

# sequential
envs = [
    gym.make("gym-generals-rllib-v0", npc=npc, agent=agent) for _ in range(10)
]
start = time.time()
t = 0
for env in tqdm.tqdm(envs):
    observation, info = env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        t += 1
    print(t)
print("Sequential time:", time.time() - start)

envs = gym.vector.AsyncVectorEnv(
    [
        lambda: gym.make("gym-generals-rllib-v0", npc=npc, agent=agent) for _ in range(10)
    ],
)


observations, infos = envs.reset()
terminated = [False] * len(observations)
truncated = [False] * len(observations)
start = time.time()
n_finished = 0
while n_finished < 10:
    actions = [envs.single_action_space.sample() for _ in range(len(observations))]
    observations, rewards, terminated, truncated, infos = envs.step(actions)
    n_finished += sum(terminated)
print("Vectorized time:", time.time() - start)
