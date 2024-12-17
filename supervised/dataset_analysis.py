import os
import numpy as np
import json
import tqdm

new_replays = [f"all_replays/new_value/{name}" for name in os.listdir("all_replays/new_value/")]
old_replays = [f"all_replays/old_value/{name}" for name in os.listdir("all_replays/old_value/")]

all_replays = new_replays + old_replays

afk_times = []
total_frames = 0
zero_moves = 0
above_2k = 0
for replay in tqdm.tqdm(all_replays):
    game = json.load(open(replay))
    if len(game["afks"]) > 0:
        if "index" in game["afks"][0]:
            time = game["afks"][0]["turn"]
        else:
            time = game["afks"][0][1]
        if time > 1500 and time < 2000:
            print(replay)
        afk_times.append(time)
    if len(game["moves"]) > 0:
        total_frames += game["moves"][-1][4]
    else:
        zero_moves += 1



afk_times = np.array(afk_times)
print(f"Total frames: {total_frames}")
print(min(afk_times), max(afk_times), np.mean(afk_times), np.median(afk_times))
print(zero_moves)
print(above_2k)
