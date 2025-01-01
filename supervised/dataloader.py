from generals.envs import PettingZooGenerals
from generals.agents import RandomAgent
from generals.core.action import compute_valid_move_mask
from neuro_agent import NeuroAgent
import json
import torch
import math
import numpy as np


class ReplayDataset(torch.utils.data.IterableDataset):
    def __init__(self, replay_files: list[str], buffer_size: int = 1000):
        self.replay_files = replay_files
        self.replays = None
        self.env = None

        self.buffer = [
            [
                np.empty((31, 24, 24), dtype=np.float32),  # Obs
                np.empty((24, 24, 4), dtype=np.float32),  # Mask
                0.0,  # Value
                np.empty(5, dtype=np.int32),  # Action
            ]
            for _ in range(buffer_size)
        ]
        self.current_replay = None
        self.buffer_idx, self.replay_idx = 0, 0
        self.filled = False
        self.A, self.B = NeuroAgent(id="red"), NeuroAgent(id="blue")

    def check_validity(self, obs, mask, action):
        # Check if the action is valid, return False if not
        _, i, j, d, _ = action
        army = obs[1][i][j]
        return d == 4 or (mask[i][j][d] == 1 and army > 1)

    def save_sample(self, obs, mask, value, action):
        self.buffer[self.buffer_idx][0][:] = obs
        self.buffer[self.buffer_idx][1][:] = mask
        self.buffer[self.buffer_idx][2] = value
        self.buffer[self.buffer_idx][3][:] = action
        self.buffer_idx += 1
        if self.buffer_idx == len(self.buffer):
            self.filled = True
            self.buffer_idx = 0
            np.random.shuffle(self.buffer)

    def teacher_action(self, moves, base, timestep):
        # throw indicates whether action should be used in training
        if timestep in moves:
            action = moves[timestep]
            throw = False
        else:
            action = [1, base[0], base[1], 4, 0]
            throw = True
        return action, throw

    def __iter__(self):
        a, b = self.A.id, self.B.id
        map, moves, values, bases, length = self.get_new_replay()
        obs, _ = self.env.reset(options={"grid": map})
        while True:
            if self.filled:
                while self.buffer_idx < len(self.buffer):
                    yield self.buffer[self.buffer_idx]
                    self.buffer_idx += 1
                self.buffer_idx = 0
                self.filled = False
            timestep = obs[a]["timestep"]
            # Take teacher actions
            a0, throw0 = self.teacher_action(moves[0], bases[0], timestep)
            a1, throw1 = self.teacher_action(moves[1], bases[1], timestep)
            actions = {a: a0, b: a1}
            # Ignore these actions, just process observations
            _ = self.A.act(obs[a])
            _ = self.B.act(obs[b])
            obs_a = self.A.last_observation
            obs_b = self.B.last_observation
            mask_a = compute_valid_move_mask(obs[a])
            mask_b = compute_valid_move_mask(obs[b])
            # Save valid observation/action pairs
            if (
                self.check_validity(obs_a, mask_a, actions[a])
                and timestep > 21
                and not throw0
            ):
                self.save_sample(obs_a, mask_a, values[0], actions[a])
            if (
                self.check_validity(obs_b, mask_b, actions[b])
                and timestep > 21
                and not throw1
            ):
                self.save_sample(obs_b, mask_b, values[1], actions[b])
            obs, _, terminated, _, _ = self.env.step(actions)
            if all(terminated.values()) or timestep >= length:
                map, moves, values, bases, length = self.get_new_replay()
                obs, _ = self.env.reset(options={"grid": map})
                self.A.reset()
                self.B.reset()

    def get_new_replay(self):
        game = json.load(open(self.replays[self.replay_idx], "r"))
        self.current_replay = self.replays[self.replay_idx]
        width = game["mapWidth"]
        height = game["mapHeight"]

        player_moves = [{}, {}]
        player_values = [-1.0, -1.0]

        # Determine winner
        if len(game["afks"]) == 0:
            winner = game["moves"][-1][0]
        else:
            winner = 1 - game["afks"][0][0]
        player_values[winner] = 1.0

        for move in game["moves"]:
            index, i, j, is50, turn = move[0], move[1], move[2], move[3], move[4]
            i, j = divmod(move[1], width)
            if move[2] == move[1] + 1:
                direction = 3
            elif move[2] == move[1] - 1:
                direction = 2
            elif move[2] == move[1] + game["mapWidth"]:
                direction = 1
            elif move[2] == move[1] - game["mapWidth"]:
                direction = 0
            player_moves[index][turn] = [0, i, j, direction, is50]

        # calculate game length as time of the last move
        game_length = game["moves"][-1][4]
        map = ["." for _ in range(width * height)]

        # place cities
        for pos, value in zip(game["cities"], game["cityArmies"]):
            map[pos] = str(value - 40) if value != 50 else "x"

        # place mountains
        for pos in game["mountains"]:
            map[pos] = "#"

        # place generals
        generals = game["generals"]
        map[generals[0]] = "A"
        map[generals[1]] = "B"
        player_bases = [
            divmod(generals[0], game["mapWidth"]),
            divmod(generals[1], game["mapWidth"]),
        ]

        # convert to 2D array
        map = [
            map[i : i + game["mapWidth"]] for i in range(0, len(map), game["mapWidth"])
        ]

        # Pad the game with '#' to make it 24x24
        pad_width = 24 - width
        pad_height = 24 - height
        map = [[*row, *["#" for _ in range(pad_width)]] for row in map]
        map.extend([["#" for _ in range(24)] for _ in range(pad_height)])

        map_str = "\n".join(["".join(row) for row in map])
        self.replay_idx += 1
        self.replay_idx %= len(self.replays)
        return (map_str, player_moves, player_values, player_bases, game_length)


def per_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    length = len(dataset.replay_files)
    per_worker = int(math.ceil((length / float(worker_info.num_workers))))

    worker_id = worker_info.id
    start = worker_id * per_worker
    end = min(start + per_worker, length)

    dataset.replays = dataset.replay_files[start:end]
    print(f"Worker {worker_id} handling replays {start} to {end}")
    print(f"Worker {worker_id} {dataset.replays[:3]}")

    dataset.env = PettingZooGenerals(
        agents={
            "red": RandomAgent("red"),
            "blue": RandomAgent("blue"),
        },
    )


def collate_fn(batch):
    observations = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
    masks = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
    values = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32)
    actions = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.int64)
    return observations, masks, values, actions
