import os
import numpy as np
import json
import tqdm

total_frames = 0
total = []
bins = [0] * 11
t = 0

# read all_replays/old/
old_replays = [f"all_replays/old/{id}" for id in os.listdir("all_replays/old/")]
new_replays = [f"all_replays/new/{id}" for id in os.listdir("all_replays/new/")]
print("Old replays: ", len(old_replays))
print("New replays: ", len(new_replays))
all_replays = old_replays + new_replays

for replay in tqdm.tqdm(all_replays):
    game = json.load(open(replay))
    frames = game["moves"][-1][4]
    total.append(frames)
    if frames == 1000:
        t+=1
    bins[frames // 100] += 1
    total_frames += frames

total = np.array(total)
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
