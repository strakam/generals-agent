import numpy as np
import time
from generals.envs import PettingZooGenerals

from generals import Replay
from generals import GridFactory
from replay_agent import ReplayAgent
import json

# index, start, end, is50, turn
game = json.load(open("supervised/all_replays/new/SlN5kByLn", "r"))
# make string long replay game["mapHeight"] * game["mapWidth"] long with "."
map = ["." for _ in range(game["mapHeight"] * game["mapWidth"])]
for pos, value in zip(game["cities"], game["cityArmies"]):
    if value == 50:
        map[pos] = "x"
    else:
        map[pos] = str(value - 40)
for pos in game["mountains"]:
    map[pos] = "#"
generals = game["generals"]
map[generals[0]] = "A"
map[generals[1]] = "B"

map = [map[i : i + game["mapWidth"]] for i in range(0, len(map), game["mapWidth"])]
map_str = "\n".join(["".join(row) for row in map])
options = {"grid": map_str, "replay_file": "verify"}


gf = GridFactory(
    mountain_density=0.2,  # Probability of a mountain in a cell
    city_density=0.05,  # Probability of a city in a cell
    general_positions=[(0, 0), (3, 3)],  # Positions of generals (i, j)
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

# Initialize agents
agent = ReplayAgent(
    game["usernames"][0], color=(255, 0, 0), replay_moves=players_moves[0]
)
npc = ReplayAgent(
    game["usernames"][1], color=(0, 0, 255), replay_moves=players_moves[1]
)

agents = {
    game["usernames"][0]: agent,
    game["usernames"][1]: npc,
}
game_length = moves[-1][4] + 1
# Create environment
env = PettingZooGenerals(
    agents=agents,
    grid_factory=gf,
    render_mode=None,
    truncation=game_length,
)
# time length of one game
start = time.time()
t = 0
print(map_str)
observations, info = env.reset(options=options)
done = False
while not done:
    actions = {}
    for k, v in observations.items():
        actions[k] = agents[k].act(v)
    observations, reward, terminated, truncated, info = env.step(actions)
    env.render()
    t += 1
    done = any(terminated.values()) or any(truncated.values())
    # env.render()
print(f"Time taken {time.time() - start:.5f} seconds")
print(t)

replay = Replay.load("verify")
replay.play()
