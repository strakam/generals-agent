import gymnasium as gym
import numpy as np

from generals import Replay
from generals.agents import ExpanderAgent
from generals import GridFactory
from replay_agent import ReplayAgent
import json

# index, start, end, is50, turn

game = json.load(open("all_replays/new/HfmB3nmWk", "r"))
# make string long replay game["mapHeight"] * game["mapWidth"] long with "." 
map = ["." for _ in range(game["mapHeight"] * game["mapWidth"])]
for pos, value in zip(game["cities"], game["cityArmies"]):
    map[pos] = str(value - 40)
for pos in game["mountains"]:
    map[pos] = "#"
generals = game["generals"]
map[generals[0]] = "A"
map[generals[1]] = "B"

map = [map[i:i+game["mapWidth"]] for i in range(0, len(map), game["mapWidth"])]
map_str = "\n".join(["".join(row) for row in map])
options = {"grid": map_str, "replay_file": "verify"}


gf = GridFactory(
    grid_dims=(4, 4),                      # Dimensions of the grid (height, width)
    mountain_density=0.2,                  # Probability of a mountain in a cell
    city_density=0.05,                     # Probability of a city in a cell
    general_positions=[(0,0),(3,3)],       # Positions of generals (i, j)
)

moves = game["moves"]
players_moves = [{}, {}]
for move in moves:
    # convert from one number to 2D index using modulo
    i, j = divmod(move[1], game["mapWidth"])
    if move[2] == move[1] + 1:
        direction = 3
    elif move[2] == move[1] - 1:
        direction = 2
    elif move[2] == move[1] + game["mapWidth"]:
        direction = 1
    elif move[2] == move[1] - game["mapWidth"]:
        direction = 0
    players_moves[move[0]][move[4]] = {
        "pass": 0,
        "cell": np.array([i, j]),
        "direction": direction,
        "split": move[3],
    }
    print(players_moves[move[0]][move[4]])
# get size of object in bytes
import sys
def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# Initialize agents
agent = ReplayAgent(players_moves[0], game["usernames"][0], color=(0, 0, 255))
npc = ReplayAgent(players_moves[1], game["usernames"][1])

# Create environment
env = gym.make("gym-generals-v0", agent=agent, npc=npc, grid_factory=gf, render_mode="human")
# time length of one game
import time
start = time.time()
t=0

observation, info = env.reset(options=options)
terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    t+=1
    env.render()
print(f"Time taken {time.time() - start:.5f} seconds")
print(t)

replay = Replay.load("verify")
replay.play()
