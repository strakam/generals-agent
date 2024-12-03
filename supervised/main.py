import os
import json
import psutil
import tqdm
from petting_replays import PettingZooGenerals
from replay_agent import ReplayAgent

a1 = ReplayAgent(id="A", color="red")
a2 = ReplayAgent(id="B", color="blue")
# Load datasets
# "all_replays/new/" and "all_replays/old/" are the directories containing the replays

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

sample = 10
new_names = os.listdir("all_replays/new/")[:sample]
old_names = os.listdir("all_replays/old/")[:sample]


env = PettingZooGenerals(
    agents={a1.id: a1, a2.id: a2},
    replay_files=[f"all_replays/new/{name}" for name in new_names],
)

obs, info = env.reset()
a1.give_actions(info[0])
a2.give_actions(info[1])
print(info)

done = False
while not done:
    actions = {a1.id: a1.act(obs[a1.id]), a2.id: a2.act(obs[a2.id])}
    print(actions)
    obs, _, terminated, truncated, _ = env.step(actions)
    done = all(terminated.values()) or all(truncated.values())
    env.render()

