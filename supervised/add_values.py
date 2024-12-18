import os
import json
import tqdm

new_replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")]
old_replays = [f"all_replays/old/{name}" for name in os.listdir("all_replays/old/")]

old_updated = "all_replays/old_value/"

all_replays = new_replays + old_replays

# human_win = 0
# human_lose = 0
# afks = 0
# total_frames = 0
# afk_timestep = []
# zero_moves = 0
# for new_replay in tqdm.tqdm(new_replays):
#     game = json.load(open(new_replay))
#     if len(game["afks"]) > 0:
#         afk_player = game["afks"][0][0]
#         afk_time = game["afks"][0][1]
#         winner = 1 - afk_player
#         afks += 1
#         afk_timestep.append(afk_time)
#         if afk_time < 80:
#             continue

cnt = 0
for old_replay in tqdm.tqdm(old_replays):
    game = json.load(open(old_replay))
    if len(game["afks"]) > 0:
        afk_player = game["afks"][0]['index']
        afk_time = game["afks"][0]['turn']
        # edit game["afks"] 
        new_format = [[afk_player, afk_time]]
        # update replay file
        game["afks"] = new_format
    with open(old_updated + old_replay.split("/")[-1], "w") as f:
        json.dump(game, f)


# print(f"Total frames: {total_frames}")
