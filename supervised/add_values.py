import os
import json
import tqdm

new_replays = [f"all_replays/new/{name}" for name in os.listdir("all_replays/new/")]
old_replays = [f"all_replays/old/{name}" for name in os.listdir("all_replays/old/")]

all_replays = new_replays + old_replays

human_win = 0
human_lose = 0
afks = 0
afk_timestep = []
# Check who won, add his index as new value into the json and save to folder all_replays/new_value/id..
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
#     else:
#         winner = game["moves"][-1][0]
#
#     game["winner"] = winner
#     # save to new folder
#     with open(new_replay.replace("new", "new_value"), "w") as f:
#         json.dump(game, f)

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
    elif len(game["moves"]) >= 80:
        winner = game["moves"][-1][0]

    game["winner"] = winner
    if "Human.exe" in game["usernames"]:
        print('hooman')
        if game["usernames"]["winner"] == "Human.exe":
            human_win += 1
        else:
            human_lose += 1
    # save to new folder
    # with open(old_replay.replace("old", "old_value"), "w") as f:
    #     json.dump(game, f)

print(human_win, human_lose, human_win / (human_win + human_lose))
print(f"number of afks: {afks}, {afks/len(new_replays)}")
print(f"afk time: {sum(afk_timestep)/len(afk_timestep)}")
print(f"median afk time: {sorted(afk_timestep)[len(afk_timestep)//2]}")
print(f"afks under 30moves: {len([t for t in afk_timestep if t < 30])}")
print(f"afks under 80moves: {len([t for t in afk_timestep if t < 80])}")
print(f"afks under 100 moves: {len([t for t in afk_timestep if t < 100])}")
# for replay in all_replays[:30]:
#     game = json.load(open(replay))
#     print(replay)
#     print(game["afks"], game["usernames"])
#     if len(game["afks"]) > 0:
#         print("AFK")
#         continue
#
#     last_move_destination = game["moves"][-1][2]
#     if last_move_destination == game["generals"][0]:
#         print(f"Player {game["usernames"][1]} wins")
#     elif last_move_destination == game["generals"][1]:
#         print(f"Player {game["usernames"][0]} wins")
#
#
#
#


