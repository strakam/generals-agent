import os
import json
import tqdm

new_replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")]
old_replays = [f"all_replays/old/{name}" for name in os.listdir("all_replays/old/")]

all_replays = new_replays + old_replays

human_win = 0
human_lose = 0
afks = 0
total_frames = 0
afk_timestep = []
for new_replay in tqdm.tqdm(new_replays):
    game = json.load(open(new_replay))
    if len(game["afks"]) > 0:
        afk_player = game["afks"][0][0]
        afk_time = game["afks"][0][1]
        winner = 1 - afk_player
        afks += 1
        afk_timestep.append(afk_time)
        if afk_time < 80:
            continue
    total_frames += game["moves"][-1][4]

cnt = 0
for old_replay in tqdm.tqdm(old_replays):
    game = json.load(open(old_replay))
    if len(game["afks"]) > 0:
        afk_player = game["afks"][0]['index']
        afk_time = game["afks"][0]['turn']
        winner = 1 - afk_player
        afks += 1
        afk_timestep.append(afk_time)
        if afk_time < 80:
            continue
    if len(game["moves"]) > 0:
        total_frames += game["moves"][-1][4]

print(f"Total frames: {total_frames}")
