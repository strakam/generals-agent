# go over all files from "all_replays/new/" and "all_replays/old/"

import os
import tqdm
import json
import numpy as np


old = os.listdir("supervised/all_replays/old/")
new = os.listdir("supervised/all_replays/new/")

widths = []
heights = []
non_afk = 0

for file in tqdm.tqdm(new):
    game = json.load(open("supervised/all_replays/new/" + file))
    widths.append(game["mapWidth"])
    heights.append(game["mapHeight"])
    non_afk += 1 if game["afks"] == [] else 0

for file in tqdm.tqdm(old):
    game = json.load(open("supervised/all_replays/old/" + file))
    widths.append(game["mapWidth"])
    heights.append(game["mapHeight"])
    non_afk += 1 if game["afks"] == [] else 0


print(f"Non afk games: {non_afk}")
print(f"Mean width: {np.mean(widths)}")
print(f"Mean height: {np.mean(heights)}")
print()
print(f"Median width: {np.median(widths)}")
print(f"Median height: {np.median(heights)}")
print()
print(f"Max width: {np.max(widths)}")
print(f"Max height: {np.max(heights)}")
print()
print(f"Min width: {np.min(widths)}")
print(f"Min height: {np.min(heights)}")
print()
print(f"Std width: {np.std(widths)}")
print(f"Std height: {np.std(heights)}")
