import os
import numpy as np
import json
import tqdm

total_frames = 0
total = []
bins = [0] * 11
both = [0, 0, 0]
high_elo_frames = 0

# read all_replays/old/
old_replays = [f"all_replays/old/{id}" for id in os.listdir("all_replays/old/")]
new_replays = [f"all_replays/new/{id}" for id in os.listdir("all_replays/new/")]
print("Old replays: ", len(old_replays))
print("New replays: ", len(new_replays))
all_replays = old_replays + new_replays

for replay in tqdm.tqdm(all_replays):
    id = replay.split("/")[-1]
    game = json.load(open(replay))
    frames = game["moves"][-1][4]
    stars = game["stars"]
    total.append(frames)
    if stars[0] > 50 and stars[1] > 50:
        high_elo_frames += frames
        both[0] += 1
        json.dump(game, open(f"high_elo_replays/{id}", "w"))
    if stars[0] > 60 and stars[1] > 60:
        both[1] += 1
    if stars[0] > 70 and stars[1] > 70:
        both[2] += 1
    bins[frames // 100] += 1
    total_frames += frames

total = np.array(total)
print("Both: ", both)
print("High elo frames: ", high_elo_frames)
print(
    len(total),
    sum(total),
    min(total),
    max(total),
    np.mean(total),
    np.median(total),
    np.std(total),
)
for i, b in enumerate(bins):
    print(i * 100, b)
