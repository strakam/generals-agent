import os
import numpy as np
import json
import tqdm

new_replays = [
    f"all_replays/new_value/{name}" for name in os.listdir("all_replays/new_value/")
]
old_replays = [
    f"all_replays/old_value/{name}" for name in os.listdir("all_replays/old_value/")
]

all_replays = new_replays + old_replays

afk_times = []
total_frames = 0
zero_moves = 0
above = [0] * 11
removed = 0
above_2k = 0

print(len(all_replays))
for replay in tqdm.tqdm(all_replays):
    game = json.load(open(replay))
    if len(game["afks"]) > 0:
        print(replay)
        if "index" in game["afks"][0]:
            time = game["afks"][0]["turn"]
        else:
            time = game["afks"][0][1]
        afk_times.append(time)
    else:
        frames = game["moves"][-1][4]
        total_frames += frames
        above[frames // 100] += 1


afk_times = np.array(afk_times)
print(f"Total frames: {total_frames}")
print(f"Removed: {removed}")
print(
    min(afk_times),
    max(afk_times),
    np.mean(afk_times),
    np.median(afk_times),
    np.std(afk_times),
)
for i, a in enumerate(above):
    print(f"Games above {i*100}: {a}")
