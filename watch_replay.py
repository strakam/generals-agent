import argparse
import numpy as np
from generals.envs import PettingZooGenerals

from generals import Replay
from replay_agent import ReplayAgent
import json

parser = argparse.ArgumentParser()
parser.add_argument("--replay", default="SlN5kByLn", type=str, help="Replay ID")

def main(args):
    replay_id = args.replay
    game = json.load(open(f"supervised/above50/{replay_id}", "r"))
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
    env = PettingZooGenerals(
        agents=agents,
        render_mode=None,
        truncation=game_length,
    )
    observations, _ = env.reset(options=options)
    done = False
    while not done:
        actions = {}
        for k, v in observations.items():
            actions[k] = agents[k].act(v)
        observations, _, terminated, truncated, _ = env.step(actions)
        env.render()
        done = any(terminated.values()) or any(truncated.values())
    replay = Replay.load("verify")
    replay.play()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
